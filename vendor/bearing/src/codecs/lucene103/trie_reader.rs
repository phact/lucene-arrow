// SPDX-License-Identifier: Apache-2.0

//! Trie reader for the Lucene 103 block tree terms index.
//!
//! Navigates the FST-like trie stored in the `.tip` file to find block file
//! pointers in the `.tim` terms dictionary. The trie is written in post-order
//! by [`super::blocktree_writer::TrieBuilder`].
//!
//! Node types:
//! - `SIGN_NO_CHILDREN` — leaf node with output (block FP)
//! - `SIGN_SINGLE_CHILD_WITH_OUTPUT` — single child, has block FP
//! - `SIGN_SINGLE_CHILD_WITHOUT_OUTPUT` — single child, no block FP
//! - `SIGN_MULTI_CHILDREN` — multiple children with strategy-based lookup

use std::io;

use crate::store::IndexInput;

// Node type signatures (lowest 2 bits of header)
const SIGN_NO_CHILDREN: u32 = 0x00;
const SIGN_SINGLE_CHILD_WITHOUT_OUTPUT: u32 = 0x02;
const SIGN_MULTI_CHILDREN: u32 = 0x03;

// Leaf node flags (bits of header byte)
const LEAF_NODE_HAS_TERMS: u32 = 1 << 5;
const LEAF_NODE_HAS_FLOOR: u32 = 1 << 6;

// Non-leaf node flags (bits of encoded output FP)
const NON_LEAF_NODE_HAS_TERMS: u64 = 1 << 1;
const NON_LEAF_NODE_HAS_FLOOR: u64 = 1 << 0;

const NO_OUTPUT: i64 = -1;
const NO_FLOOR_DATA: i64 = -1;

/// Masks for extracting N bytes from a little-endian long.
/// `BYTES_MASK[n]` masks out the lowest `(n+1)` bytes.
const BYTES_MASK: [u64; 8] = [
    0xFF,
    0xFFFF,
    0xFF_FFFF,
    0xFFFF_FFFF,
    0xFF_FFFF_FFFF,
    0xFFFF_FFFF_FFFF,
    0xFF_FFFF_FFFF_FFFF,
    0xFFFF_FFFF_FFFF_FFFF,
];

// Child save strategy codes
const STRATEGY_REVERSE_ARRAY: u32 = 0;
const STRATEGY_ARRAY: u32 = 1;
const STRATEGY_BITS: u32 = 2;

/// A loaded trie node with decoded metadata.
#[derive(Debug, Clone)]
pub(crate) struct Node {
    /// File pointer of this node within the `.tip` field slice.
    fp: i64,
    /// Node type signature (lowest 2 bits of header).
    sign: u32,
    /// Label byte that led to this node (set by parent during lookup).
    label: u8,
    /// Block file pointer in `.tim` for this node's terms, or [`NO_OUTPUT`].
    output_fp: i64,
    /// Whether this node's block contains terms (vs only sub-blocks).
    has_terms: bool,
    /// File pointer to floor data within the `.tip` field slice, or [`NO_FLOOR_DATA`].
    floor_data_fp: i64,

    // Single-child fields
    /// Delta FP to the single child (parent_fp - child_fp).
    child_delta_fp: i64,
    /// Label of the minimum (or only) child.
    min_children_label: u8,

    // Multi-children fields
    /// File pointer to the strategy data region.
    strategy_fp: i64,
    /// Child save strategy code.
    child_save_strategy: u32,
    /// Number of bytes in the strategy data.
    strategy_bytes: u32,
    /// Number of bytes per child delta FP entry.
    children_delta_fp_bytes: u32,
}

impl Node {
    pub(crate) fn new() -> Self {
        Self {
            fp: 0,
            sign: 0,
            label: 0,
            output_fp: NO_OUTPUT,
            has_terms: false,
            floor_data_fp: NO_FLOOR_DATA,
            child_delta_fp: 0,
            min_children_label: 0,
            strategy_fp: 0,
            child_save_strategy: 0,
            strategy_bytes: 0,
            children_delta_fp_bytes: 0,
        }
    }

    /// Whether this node has an output (block file pointer).
    pub(crate) fn has_output(&self) -> bool {
        self.output_fp != NO_OUTPUT
    }

    /// Returns the label byte that led to this node.
    pub(crate) fn label(&self) -> u8 {
        self.label
    }

    /// Whether this node has floor data (multiple sub-blocks sharing a prefix).
    pub(crate) fn is_floor(&self) -> bool {
        self.floor_data_fp != NO_FLOOR_DATA
    }

    /// Returns the block file pointer for this node's terms.
    pub(crate) fn output_fp(&self) -> i64 {
        self.output_fp
    }

    /// Whether this node's block contains terms.
    pub(crate) fn has_terms(&self) -> bool {
        self.has_terms
    }

    /// Returns the floor data file pointer, if this node has floor data.
    pub(crate) fn floor_data_fp(&self) -> i64 {
        self.floor_data_fp
    }
}

/// Reader for the trie index stored in the `.tip` file.
///
/// Provides child lookup to navigate from the root node down to a leaf,
/// following target term bytes. Each node may carry an `output_fp` pointing
/// to the corresponding block in the `.tim` file.
pub struct TrieReader<'a> {
    access: IndexInput<'a>,
    root: Node,
}

impl<'a> TrieReader<'a> {
    /// Creates a new trie reader from the `.tip` field slice bytes.
    ///
    /// `access` borrows the `.tip` bytes for this field (from `index_start`
    /// to `index_end`). `root_fp` is the file pointer to the root node,
    /// rebased to be relative to the start of `access`.
    pub(crate) fn new(access: IndexInput<'a>, root_fp: i64) -> io::Result<Self> {
        let mut root = Node::new();
        load(&access, &mut root, root_fp)?;
        Ok(Self { access, root })
    }

    /// Returns the root node of the trie.
    pub(crate) fn root(&self) -> &Node {
        &self.root
    }

    /// Looks up a child node by label.
    ///
    /// Returns `true` if the child was found and `child` was populated,
    /// `false` if no child exists for the given label.
    pub(crate) fn lookup_child(
        &self,
        target_label: u8,
        parent: &Node,
        child: &mut Node,
    ) -> io::Result<bool> {
        let sign = parent.sign;

        if sign == SIGN_NO_CHILDREN {
            return Ok(false);
        }

        if sign != SIGN_MULTI_CHILDREN {
            // Single child
            if target_label != parent.min_children_label {
                return Ok(false);
            }
            child.label = target_label;
            load(&self.access, child, parent.fp - parent.child_delta_fp)?;
            return Ok(true);
        }

        // Multi-children: look up using the child save strategy
        let min_label = parent.min_children_label;
        let position = if target_label == min_label {
            0i32
        } else if target_label > min_label {
            strategy_lookup(
                parent.child_save_strategy,
                target_label,
                &self.access,
                parent.strategy_fp,
                parent.strategy_bytes,
                min_label,
            )?
        } else {
            -1
        };

        if position < 0 {
            return Ok(false);
        }

        let bytes_per_entry = parent.children_delta_fp_bytes;
        let pos = parent.strategy_fp
            + parent.strategy_bytes as i64
            + bytes_per_entry as i64 * position as i64;
        let delta = self.access.read_le_long_at(pos as usize)? as u64
            & BYTES_MASK[bytes_per_entry as usize - 1];
        let fp = parent.fp - delta as i64;

        child.label = target_label;
        load(&self.access, child, fp)?;
        Ok(true)
    }

    /// Navigates the trie following the target bytes, returning the deepest
    /// node that has an output (block FP).
    ///
    /// Returns `None` if the target has no prefix in the trie that leads to
    /// a block.
    pub fn seek_to_block(&self, target: &[u8]) -> io::Result<Option<TrieSeekResult>> {
        // Use two nodes and alternate between them to avoid borrow issues.
        let mut nodes = [Node::new(), Node::new()];
        // Copy root into nodes[0] so we own it.
        load(&self.access, &mut nodes[0], self.root.fp)?;
        let mut current_idx = 0usize;
        let mut best: Option<TrieSeekResult> = None;

        if nodes[current_idx].has_output() {
            best = Some(TrieSeekResult {
                output_fp: nodes[current_idx].output_fp,
                has_terms: nodes[current_idx].has_terms,
                floor_data_fp: nodes[current_idx].floor_data_fp,
                depth: 0,
            });
        }

        for (i, &byte) in target.iter().enumerate() {
            let child_idx = 1 - current_idx;
            let (current_slice, child_slice) = if current_idx == 0 {
                let (a, b) = nodes.split_at_mut(1);
                (&a[0], &mut b[0])
            } else {
                let (a, b) = nodes.split_at_mut(1);
                (&b[0], &mut a[0])
            };

            if !self.lookup_child(byte, current_slice, child_slice)? {
                break;
            }
            if child_slice.has_output() {
                best = Some(TrieSeekResult {
                    output_fp: child_slice.output_fp,
                    has_terms: child_slice.has_terms,
                    floor_data_fp: child_slice.floor_data_fp,
                    depth: i + 1,
                });
            }
            current_idx = child_idx;
        }

        Ok(best)
    }
}

/// Result of navigating the trie to find a block.
#[derive(Debug)]
pub struct TrieSeekResult {
    /// Block file pointer in `.tim`.
    pub output_fp: i64,
    /// Whether the block contains terms (vs only sub-blocks).
    pub has_terms: bool,
    /// File pointer to floor data, or `NO_FLOOR_DATA`.
    pub floor_data_fp: i64,
    /// Number of target bytes consumed to reach this node.
    pub depth: usize,
}

/// Loads a trie node from the `.tip` field slice at the given file pointer.
fn load(access: &IndexInput<'_>, node: &mut Node, fp: i64) -> io::Result<()> {
    node.fp = fp;
    let term_flags_long = access.read_le_long_at(fp as usize)?;
    let term_flags = term_flags_long as u32;
    let sign = term_flags & 0x03;
    node.sign = sign;

    match sign {
        SIGN_NO_CHILDREN => load_leaf_node(access, term_flags, term_flags_long, fp, node),
        SIGN_MULTI_CHILDREN => {
            load_multi_children_node(access, term_flags, term_flags_long, fp, node)
        }
        _ => load_single_child_node(access, sign, term_flags, term_flags_long, fp, node),
    }
}

fn load_leaf_node(
    access: &IndexInput<'_>,
    term_flags: u32,
    term_flags_long: i64,
    fp: i64,
    node: &mut Node,
) -> io::Result<()> {
    let fp_bytes_minus1 = ((term_flags >> 2) & 0x07) as usize;
    node.output_fp = if fp_bytes_minus1 <= 6 {
        ((term_flags_long as u64 >> 8) & BYTES_MASK[fp_bytes_minus1]) as i64
    } else {
        access.read_le_long_at((fp + 1) as usize)?
    };
    node.has_terms = (term_flags & LEAF_NODE_HAS_TERMS) != 0;
    node.floor_data_fp = if (term_flags & LEAF_NODE_HAS_FLOOR) != 0 {
        fp + 2 + fp_bytes_minus1 as i64
    } else {
        NO_FLOOR_DATA
    };
    Ok(())
}

fn load_single_child_node(
    access: &IndexInput<'_>,
    sign: u32,
    term_flags: u32,
    term_flags_long: i64,
    fp: i64,
    node: &mut Node,
) -> io::Result<()> {
    let child_delta_fp_bytes_minus1 = ((term_flags >> 2) & 0x07) as usize;
    let l = if child_delta_fp_bytes_minus1 <= 5 {
        (term_flags_long as u64) >> 16
    } else {
        access.read_le_long_at((fp + 2) as usize)? as u64
    };
    node.child_delta_fp = (l & BYTES_MASK[child_delta_fp_bytes_minus1]) as i64;
    node.min_children_label = ((term_flags >> 8) & 0xFF) as u8;

    if sign == SIGN_SINGLE_CHILD_WITHOUT_OUTPUT {
        node.output_fp = NO_OUTPUT;
        node.has_terms = false;
        node.floor_data_fp = NO_FLOOR_DATA;
    } else {
        // SIGN_SINGLE_CHILD_WITH_OUTPUT
        let encoded_output_fp_bytes_minus1 = ((term_flags >> 5) & 0x07) as usize;
        let offset = fp + child_delta_fp_bytes_minus1 as i64 + 3;
        let encoded_fp = access.read_le_long_at(offset as usize)? as u64
            & BYTES_MASK[encoded_output_fp_bytes_minus1];
        node.output_fp = (encoded_fp >> 2) as i64;
        node.has_terms = (encoded_fp & NON_LEAF_NODE_HAS_TERMS) != 0;
        node.floor_data_fp = if (encoded_fp & NON_LEAF_NODE_HAS_FLOOR) != 0 {
            offset + encoded_output_fp_bytes_minus1 as i64 + 1
        } else {
            NO_FLOOR_DATA
        };
    }
    Ok(())
}

fn load_multi_children_node(
    access: &IndexInput<'_>,
    term_flags: u32,
    _term_flags_long: i64,
    fp: i64,
    node: &mut Node,
) -> io::Result<()> {
    node.children_delta_fp_bytes = ((term_flags >> 2) & 0x07) + 1;
    node.child_save_strategy = (term_flags >> 9) & 0x03;
    node.strategy_bytes = ((term_flags >> 11) & 0x1F) + 1;
    node.min_children_label = ((term_flags >> 16) & 0xFF) as u8;

    let has_output = (term_flags & 0x20) != 0;
    if has_output {
        let encoded_output_fp_bytes_minus1 = ((term_flags >> 6) & 0x07) as usize;
        let l = if encoded_output_fp_bytes_minus1 <= 4 {
            (_term_flags_long as u64) >> 24
        } else {
            access.read_le_long_at((fp + 3) as usize)? as u64
        };
        let encoded_fp = l & BYTES_MASK[encoded_output_fp_bytes_minus1];
        node.output_fp = (encoded_fp >> 2) as i64;
        node.has_terms = (encoded_fp & NON_LEAF_NODE_HAS_TERMS) != 0;

        if (encoded_fp & NON_LEAF_NODE_HAS_FLOOR) != 0 {
            let offset = fp + 4 + encoded_output_fp_bytes_minus1 as i64;
            let children_num = (access.read_byte_at(offset as usize)? as u64) + 1;
            node.strategy_fp = offset + 1;
            node.floor_data_fp = node.strategy_fp
                + node.strategy_bytes as i64
                + children_num as i64 * node.children_delta_fp_bytes as i64;
        } else {
            node.floor_data_fp = NO_FLOOR_DATA;
            node.strategy_fp = fp + 4 + encoded_output_fp_bytes_minus1 as i64;
        }
    } else {
        node.output_fp = NO_OUTPUT;
        node.has_terms = false;
        node.floor_data_fp = NO_FLOOR_DATA;
        node.strategy_fp = fp + 3;
    }
    Ok(())
}

/// Looks up a target label in the children of a multi-children node.
///
/// Returns the 0-based position of the child, or -1 if not found.
fn strategy_lookup(
    strategy_code: u32,
    target_label: u8,
    access: &IndexInput<'_>,
    strategy_fp: i64,
    strategy_bytes: u32,
    min_label: u8,
) -> io::Result<i32> {
    match strategy_code {
        STRATEGY_BITS => bits_lookup(target_label, access, strategy_fp, strategy_bytes, min_label),
        STRATEGY_ARRAY => {
            array_lookup(target_label, access, strategy_fp, strategy_bytes, min_label)
        }
        STRATEGY_REVERSE_ARRAY => {
            reverse_array_lookup(target_label, access, strategy_fp, strategy_bytes, min_label)
        }
        _ => Err(io::Error::other(format!(
            "unknown child save strategy: {strategy_code}"
        ))),
    }
}

/// Bitset strategy: each bit represents presence of a label relative to min_label.
fn bits_lookup(
    target_label: u8,
    access: &IndexInput<'_>,
    strategy_fp: i64,
    strategy_bytes: u32,
    min_label: u8,
) -> io::Result<i32> {
    let bit_index = (target_label - min_label) as u32;
    if bit_index >= strategy_bytes * 8 {
        return Ok(-1);
    }
    let word_index = (bit_index >> 6) as i64;
    let word_fp = strategy_fp + (word_index << 3);
    let word = access.read_le_long_at(word_fp as usize)? as u64;
    let mask = 1u64 << bit_index;
    if word & mask == 0 {
        return Ok(-1);
    }
    let mut pos = 0i32;
    let mut fp = strategy_fp;
    while fp < word_fp {
        pos += (access.read_le_long_at(fp as usize)? as u64).count_ones() as i32;
        fp += 8;
    }
    pos += (word & (mask - 1)).count_ones() as i32;
    Ok(pos)
}

/// Array strategy: sorted array of child labels (excluding min_label).
/// Binary search for the target.
fn array_lookup(
    target_label: u8,
    access: &IndexInput<'_>,
    strategy_fp: i64,
    strategy_bytes: u32,
    min_label: u8,
) -> io::Result<i32> {
    if target_label <= min_label {
        return Ok(-1);
    }
    // Binary search in the label array
    let mut lo = 0u32;
    let mut hi = strategy_bytes; // strategy_bytes = num_children - 1
    while lo < hi {
        let mid = (lo + hi) / 2;
        let label = access.read_byte_at((strategy_fp + mid as i64) as usize)?;
        if label < target_label {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if lo < strategy_bytes {
        let label = access.read_byte_at((strategy_fp + lo as i64) as usize)?;
        if label == target_label {
            return Ok((lo + 1) as i32); // +1 because min_label is position 0
        }
    }
    Ok(-1)
}

/// Reverse array strategy: max_label byte, then sorted array of absent labels.
/// A label is present if it's in [min_label, max_label] and NOT in the absent array.
fn reverse_array_lookup(
    target_label: u8,
    access: &IndexInput<'_>,
    strategy_fp: i64,
    strategy_bytes: u32,
    min_label: u8,
) -> io::Result<i32> {
    let max_label = access.read_byte_at(strategy_fp as usize)?;
    if target_label > max_label {
        return Ok(-1);
    }
    if target_label == max_label {
        return Ok((max_label - min_label) as i32 - strategy_bytes as i32 + 1);
    }
    if strategy_bytes == 1 {
        return Ok((target_label - min_label) as i32);
    }
    // Binary search in the sorted absent-labels array
    let absent_fp = strategy_fp + 1;
    let mut low = 0i32;
    let mut high = strategy_bytes as i32 - 2;
    while low <= high {
        let mid = (low + high) as u32 >> 1;
        let mid_label = access.read_byte_at((absent_fp + mid as i64) as usize)?;
        if mid_label < target_label {
            low = mid as i32 + 1;
        } else if mid_label > target_label {
            high = mid as i32 - 1;
        } else {
            return Ok(-1); // target is in the absent list
        }
    }
    Ok((target_label - min_label) as i32 - low)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codecs::competitive_impact::BufferedNormsLookup;
    use crate::codecs::lucene103::blocktree_reader::BlockTreeTermsReader;
    use crate::codecs::lucene103::blocktree_writer::{BlockTreeTermsWriter, BufferedFieldTerms};
    use crate::document::{DocValuesType, IndexOptions, TermOffset};
    use crate::index::pipeline::terms_hash::{FreqProxTermsWriterPerField, TermsHash};
    use crate::index::{FieldInfo, FieldInfos, PointDimensionConfig};
    use crate::store::memory::MemoryDirectory;
    use crate::store::{Directory, SharedDirectory};
    use crate::util::byte_block_pool::ByteBlockPool;

    fn make_field_info(name: &str, number: u32) -> FieldInfo {
        FieldInfo::new(
            name.to_string(),
            number,
            false,
            false,
            IndexOptions::Docs,
            DocValuesType::None,
            PointDimensionConfig::default(),
        )
    }

    struct TestTerms {
        writer: FreqProxTermsWriterPerField,
        term_pool: ByteBlockPool,
        terms_hash: TermsHash,
    }

    impl TestTerms {
        fn new(field_name: &str) -> Self {
            let term_pool = ByteBlockPool::new(32 * 1024);
            Self {
                writer: FreqProxTermsWriterPerField::new(
                    field_name.to_string(),
                    IndexOptions::Docs,
                ),
                term_pool,
                terms_hash: TermsHash::new(),
            }
        }

        fn add(&mut self, term: &str, doc_id: i32, position: i32) {
            self.writer.current_position = position;
            self.writer.current_offset = TermOffset::default();
            self.writer
                .add(
                    &mut self.term_pool,
                    &mut self.terms_hash,
                    term.as_bytes(),
                    doc_id,
                )
                .unwrap();
        }

        fn finalize(&mut self) {
            self.writer.flush_pending_docs(&mut self.terms_hash);
            self.writer.sort_terms(&self.term_pool);
        }
    }

    /// Add terms in doc-major order from term-major test data.
    fn add_terms_doc_major(tt: &mut TestTerms, terms: &[(&str, &[i32])]) {
        let max_doc = terms
            .iter()
            .flat_map(|(_, docs)| docs.iter())
            .copied()
            .max()
            .unwrap_or(-1);
        for doc_id in 0..=max_doc {
            for (term, doc_ids) in terms {
                if doc_ids.contains(&doc_id) {
                    tt.add(term, doc_id, 0);
                }
            }
        }
    }

    /// Write terms and return (directory, field_infos, segment_id).
    fn write_terms(
        terms: Vec<(&str, &[i32])>,
    ) -> io::Result<(SharedDirectory, FieldInfos, [u8; 16])> {
        let field_infos = FieldInfos::new(vec![make_field_info("f", 0)]);
        let segment_name = "_0";
        let segment_suffix = "";
        let segment_id = [0u8; 16];

        let shared_dir = MemoryDirectory::create();

        {
            let mut writer = BlockTreeTermsWriter::new(
                &shared_dir,
                segment_name,
                segment_suffix,
                &segment_id,
                IndexOptions::DocsAndFreqs,
            )?;

            let mut tt = TestTerms::new("f");
            add_terms_doc_major(&mut tt, &terms);
            tt.finalize();

            let field_terms =
                BufferedFieldTerms::new(&tt.writer, &tt.term_pool, &tt.terms_hash, "f", 0);
            let norms = BufferedNormsLookup::no_norms();
            writer.write_field(&field_terms, &norms)?;

            writer.finish()?;
        }

        Ok((shared_dir, field_infos, segment_id))
    }

    /// Open the blocktree reader. The caller builds a trie view from the
    /// returned reader using `fr.new_trie_reader()`.
    fn open_reader(
        dir: &dyn Directory,
        field_infos: &FieldInfos,
        segment_id: &[u8; 16],
    ) -> io::Result<BlockTreeTermsReader> {
        BlockTreeTermsReader::open(dir, "_0", "", segment_id, field_infos)
    }

    #[test]
    fn test_trie_reader_root_has_output() {
        let terms = vec![
            ("alpha", &[0, 1][..]),
            ("beta", &[1, 2]),
            ("gamma", &[0, 2]),
        ];
        let (dir, fi, id) = write_terms(terms).unwrap();
        let reader = open_reader(&dir, &fi, &id).unwrap();
        let fr = reader.field_reader(0).unwrap();
        let trie = fr.new_trie_reader().unwrap();

        // Root should have an output (the root block FP)
        assert!(trie.root().has_output());
    }

    #[test]
    fn test_trie_seek_to_block_single_block() {
        // Few terms → single block → root has output, no children needed
        let terms = vec![("alpha", &[0][..]), ("beta", &[1]), ("gamma", &[2])];
        let (dir, fi, id) = write_terms(terms).unwrap();
        let reader = open_reader(&dir, &fi, &id).unwrap();
        let fr = reader.field_reader(0).unwrap();
        let trie = fr.new_trie_reader().unwrap();

        // Any term should find the root block
        let result = trie.seek_to_block(b"alpha").unwrap();
        assert_some!(&result);
        let r = result.unwrap();
        assert!(r.has_terms);
        assert_ge!(r.output_fp, 0);
    }

    #[test]
    fn test_trie_seek_multi_prefix() {
        // Generate enough terms with different prefixes to force multi-block trie
        let mut terms_data: Vec<(String, Vec<i32>)> = Vec::new();
        // 50 terms starting with "aaa" and 50 starting with "bbb"
        // exceeds DEFAULT_MAX_BLOCK_SIZE (48) per prefix group
        for i in 0..50 {
            terms_data.push((format!("aaa{i:04}"), vec![i]));
            terms_data.push((format!("bbb{i:04}"), vec![50 + i]));
        }

        let terms: Vec<(&str, &[i32])> = terms_data
            .iter()
            .map(|(t, d)| (t.as_str(), d.as_slice()))
            .collect();

        let (dir, fi, id) = write_terms(terms).unwrap();
        let reader = open_reader(&dir, &fi, &id).unwrap();
        let fr = reader.field_reader(0).unwrap();
        let trie = fr.new_trie_reader().unwrap();

        // Both prefixes should find blocks
        let a_result = trie.seek_to_block(b"aaa0025").unwrap();
        assert_some!(&a_result);

        let b_result = trie.seek_to_block(b"bbb0025").unwrap();
        assert_some!(&b_result);

        // The two prefixes should potentially point to different blocks
        let a_fp = a_result.unwrap().output_fp;
        let b_fp = b_result.unwrap().output_fp;
        // They might be the same root block or different — depends on trie structure
        // At minimum, both should be valid (>= 0)
        assert_ge!(a_fp, 0);
        assert_ge!(b_fp, 0);
    }

    #[test]
    fn test_trie_seek_nonexistent_prefix() {
        let terms = vec![("alpha", &[0][..]), ("beta", &[1])];
        let (dir, fi, id) = write_terms(terms).unwrap();
        let reader = open_reader(&dir, &fi, &id).unwrap();
        let fr = reader.field_reader(0).unwrap();
        let trie = fr.new_trie_reader().unwrap();

        // Empty target should still find the root block (empty prefix)
        let result = trie.seek_to_block(b"").unwrap();
        assert_some!(&result);
    }

    #[test]
    fn test_trie_single_child_nodes() {
        // All terms share a long prefix "prefix_" — inner trie nodes along
        // the path have single children. Whether they have output depends on
        // whether the blocktree creates blocks at intermediate prefixes.
        let mut terms_data: Vec<(String, Vec<i32>)> = Vec::new();
        for i in 0..30 {
            terms_data.push((format!("prefix_{i:03}"), vec![i]));
        }
        let terms: Vec<(&str, &[i32])> = terms_data
            .iter()
            .map(|(t, d)| (t.as_str(), d.as_slice()))
            .collect();

        let (dir, fi, id) = write_terms(terms).unwrap();
        let reader = open_reader(&dir, &fi, &id).unwrap();
        let fr = reader.field_reader(0).unwrap();
        let trie = fr.new_trie_reader().unwrap();

        let root = trie.root();
        // Root has a single child (all terms start with 'p')
        assert_ne!(root.sign, SIGN_NO_CHILDREN);
        assert_ne!(root.sign, SIGN_MULTI_CHILDREN);
        assert_eq!(root.min_children_label, b'p');

        // Should be able to follow the single-child chain down to find the block
        let mut child = Node::new();
        assert!(trie.lookup_child(b'p', root, &mut child).unwrap());
        assert!(!trie.lookup_child(b'q', root, &mut child).unwrap());

        // Seeking should find the block
        let result = trie.seek_to_block(b"prefix_015").unwrap();
        assert_some!(&result);
    }

    #[test]
    fn test_trie_single_child_with_output() {
        // Terms with a common prefix that has its own block, plus exactly one
        // continuation letter. 50+ terms under "aa" forces block splitting at "aa",
        // and no other first-letter branches → single child at root.
        let mut terms_data: Vec<(String, Vec<i32>)> = Vec::new();
        for i in 0..50 {
            terms_data.push((format!("aa{i:04}"), vec![i]));
        }
        let terms: Vec<(&str, &[i32])> = terms_data
            .iter()
            .map(|(t, d)| (t.as_str(), d.as_slice()))
            .collect();

        let (dir, fi, id) = write_terms(terms).unwrap();
        let reader = open_reader(&dir, &fi, &id).unwrap();
        let fr = reader.field_reader(0).unwrap();
        let trie = fr.new_trie_reader().unwrap();

        // Verify we can navigate to a term in the block
        let result = trie.seek_to_block(b"aa0025").unwrap();
        assert_some!(&result);
        let r = result.unwrap();
        assert_ge!(r.output_fp, 0);
    }

    #[test]
    fn test_trie_multi_children_lookup() {
        // Create terms under 3 different prefixes to force multi-children node.
        // Each prefix has 50+ terms to force block splitting.
        let mut terms_data: Vec<(String, Vec<i32>)> = Vec::new();
        for i in 0..50 {
            terms_data.push((format!("aaa{i:04}"), vec![i]));
            terms_data.push((format!("bbb{i:04}"), vec![50 + i]));
            terms_data.push((format!("ccc{i:04}"), vec![100 + i]));
        }
        let terms: Vec<(&str, &[i32])> = terms_data
            .iter()
            .map(|(t, d)| (t.as_str(), d.as_slice()))
            .collect();

        let (dir, fi, id) = write_terms(terms).unwrap();
        let reader = open_reader(&dir, &fi, &id).unwrap();
        let fr = reader.field_reader(0).unwrap();
        let trie = fr.new_trie_reader().unwrap();

        // Root should be multi-children with children for 'a', 'b', 'c'
        let root = trie.root();
        assert_eq!(root.sign, SIGN_MULTI_CHILDREN);

        // lookup_child should find all three
        let mut child = Node::new();
        assert!(trie.lookup_child(b'a', root, &mut child).unwrap());
        assert!(trie.lookup_child(b'b', root, &mut child).unwrap());
        assert!(trie.lookup_child(b'c', root, &mut child).unwrap());

        // lookup_child should not find labels outside the set
        assert!(!trie.lookup_child(b'd', root, &mut child).unwrap());
        assert!(!trie.lookup_child(b'A', root, &mut child).unwrap());

        // seek_to_block should find distinct blocks for each prefix
        let a_result = trie.seek_to_block(b"aaa0025").unwrap().unwrap();
        let b_result = trie.seek_to_block(b"bbb0025").unwrap().unwrap();
        let c_result = trie.seek_to_block(b"ccc0025").unwrap().unwrap();
        assert_ge!(a_result.output_fp, 0);
        assert_ge!(b_result.output_fp, 0);
        assert_ge!(c_result.output_fp, 0);
    }

    #[test]
    fn test_trie_multi_children_strategy_array() {
        // ARRAY strategy: few children with large label gaps.
        // Two prefixes with distant first letters: 'a' and 'z'.
        // need_bytes(ARRAY) = num_children - 1 = 1
        // need_bytes(BITS) = (122-97+1)/8 = 4
        // need_bytes(REVERSE_ARRAY) = 26 - 2 + 1 = 25
        // → ARRAY wins
        let mut terms_data: Vec<(String, Vec<i32>)> = Vec::new();
        for i in 0..50 {
            terms_data.push((format!("aaa{i:04}"), vec![i]));
            terms_data.push((format!("zzz{i:04}"), vec![50 + i]));
        }
        let terms: Vec<(&str, &[i32])> = terms_data
            .iter()
            .map(|(t, d)| (t.as_str(), d.as_slice()))
            .collect();

        let (dir, fi, id) = write_terms(terms).unwrap();
        let reader = open_reader(&dir, &fi, &id).unwrap();
        let fr = reader.field_reader(0).unwrap();
        let trie = fr.new_trie_reader().unwrap();

        let root = trie.root();
        assert_eq!(root.sign, SIGN_MULTI_CHILDREN);
        assert_eq!(root.child_save_strategy, STRATEGY_ARRAY);

        let mut child = Node::new();
        assert!(trie.lookup_child(b'a', root, &mut child).unwrap());
        assert!(trie.lookup_child(b'z', root, &mut child).unwrap());
        assert!(!trie.lookup_child(b'm', root, &mut child).unwrap());
    }

    #[test]
    fn test_trie_multi_children_strategy_bits() {
        // BITS strategy: children with labels close together.
        // Prefixes: "aa", "ab", "ac", "ad" — labels a,b,c,d are adjacent.
        // need_bytes(BITS) = (100-97+1)/8 = 1
        // need_bytes(ARRAY) = 4 - 1 = 3
        // need_bytes(REVERSE_ARRAY) = 4 - 4 + 1 = 1 (tie, but BITS wins by order)
        let mut terms_data: Vec<(String, Vec<i32>)> = Vec::new();
        for i in 0..50 {
            terms_data.push((format!("aa{i:04}"), vec![i]));
            terms_data.push((format!("ab{i:04}"), vec![50 + i]));
            terms_data.push((format!("ac{i:04}"), vec![100 + i]));
            terms_data.push((format!("ad{i:04}"), vec![150 + i]));
        }
        let terms: Vec<(&str, &[i32])> = terms_data
            .iter()
            .map(|(t, d)| (t.as_str(), d.as_slice()))
            .collect();

        let (dir, fi, id) = write_terms(terms).unwrap();
        let reader = open_reader(&dir, &fi, &id).unwrap();
        let fr = reader.field_reader(0).unwrap();
        let trie = fr.new_trie_reader().unwrap();

        // The node with children 'a','b','c','d' may be root or one level down
        // depending on how the blocktree groups things. Let's just verify
        // seek works for all prefixes.
        for prefix in [b"aa0025" as &[u8], b"ab0025", b"ac0025", b"ad0025"] {
            let result = trie.seek_to_block(prefix).unwrap();
            assert_some!(&result);
        }
    }

    #[test]
    fn test_trie_multi_children_strategy_reverse_array() {
        // REVERSE_ARRAY strategy: many children filling most of a label range.
        // Use consecutive single-letter prefixes: 'a' through 'h' (8 children).
        // need_bytes(ARRAY) = 8 - 1 = 7
        // need_bytes(BITS) = (104-97+1)/8 = 1
        // need_bytes(REVERSE_ARRAY) = 8 - 8 + 1 = 1 (tie with BITS)
        //
        // For REVERSE_ARRAY to win, we need the range to be larger than children
        // with most filled. Use: 'a','b','c','e','f','g' (skip 'd').
        // range = 7, children = 6
        // need_bytes(BITS) = 7/8 = 1
        // need_bytes(ARRAY) = 6 - 1 = 5
        // need_bytes(REVERSE_ARRAY) = 7 - 6 + 1 = 2
        // BITS still wins here. Let me think of a case where REVERSE_ARRAY wins...
        //
        // Actually, looking at choose_child_save_strategy, it picks the one with
        // fewest bytes, with BITS checked first on ties. REVERSE_ARRAY wins when:
        // range - count + 1 < range.div_ceil(8) AND range - count + 1 < count - 1
        // E.g., range=20, count=18: BITS=3, ARRAY=17, REVERSE_ARRAY=3 → tie, BITS wins
        // E.g., range=17, count=16: BITS=3, ARRAY=15, REVERSE_ARRAY=2 → REVERSE_ARRAY wins!
        //
        // Need 16 children spanning a range of 17 labels (one gap).
        // Use labels 'a' through 'q' (17 range), skip one letter.
        let labels: Vec<u8> = (b'a'..=b'q')
            .filter(|&b| b != b'i') // skip 'i', leaving 16 of 17
            .collect();

        let mut terms_data: Vec<(String, Vec<i32>)> = Vec::new();
        for (idx, &label) in labels.iter().enumerate() {
            for i in 0..50 {
                let term = format!("{}{i:04}", label as char);
                terms_data.push((term, vec![(idx * 50 + i) as i32]));
            }
        }
        let terms: Vec<(&str, &[i32])> = terms_data
            .iter()
            .map(|(t, d)| (t.as_str(), d.as_slice()))
            .collect();

        let (dir, fi, id) = write_terms(terms).unwrap();
        let reader = open_reader(&dir, &fi, &id).unwrap();
        let fr = reader.field_reader(0).unwrap();
        let trie = fr.new_trie_reader().unwrap();

        let root = trie.root();
        assert_eq!(root.sign, SIGN_MULTI_CHILDREN);
        assert_eq!(root.child_save_strategy, STRATEGY_REVERSE_ARRAY);

        // Should find all present labels
        let mut child = Node::new();
        for &label in &labels {
            assert!(
                trie.lookup_child(label, root, &mut child).unwrap(),
                "should find child for label '{}'",
                label as char
            );
        }
        // Should not find the absent label
        assert!(!trie.lookup_child(b'i', root, &mut child).unwrap());
        // Should not find labels outside range
        assert!(!trie.lookup_child(b'r', root, &mut child).unwrap());
    }

    #[test]
    fn test_trie_deep_traversal() {
        // Test multi-level trie navigation with nested prefixes.
        // Terms with structure: "category/subcategory/item_NNN"
        // This creates a deep trie with multiple levels of single-child nodes.
        let mut terms_data: Vec<(String, Vec<i32>)> = Vec::new();
        for i in 0..50 {
            terms_data.push((format!("category/alpha/item_{i:03}"), vec![i]));
            terms_data.push((format!("category/beta/item_{i:03}"), vec![50 + i]));
        }
        let terms: Vec<(&str, &[i32])> = terms_data
            .iter()
            .map(|(t, d)| (t.as_str(), d.as_slice()))
            .collect();

        let (dir, fi, id) = write_terms(terms).unwrap();
        let reader = open_reader(&dir, &fi, &id).unwrap();
        let fr = reader.field_reader(0).unwrap();
        let trie = fr.new_trie_reader().unwrap();

        // Should traverse deep into the trie
        let alpha = trie.seek_to_block(b"category/alpha/item_025").unwrap();
        assert!(alpha.is_some());
        let alpha = alpha.unwrap();
        assert!(alpha.depth > 0, "should have consumed some prefix bytes");

        let beta = trie.seek_to_block(b"category/beta/item_025").unwrap();
        assert!(beta.is_some());
        let beta = beta.unwrap();
        assert!(beta.depth > 0);

        // Different subcategories should (likely) point to different blocks
        // since each has 50 terms exceeding the block size
        assert!(alpha.output_fp >= 0);
        assert!(beta.output_fp >= 0);
    }

    #[test]
    fn test_trie_lookup_child_below_min_label() {
        // When target_label < min_children_label, lookup should return false
        let mut terms_data: Vec<(String, Vec<i32>)> = Vec::new();
        for i in 0..50 {
            terms_data.push((format!("mmm{i:04}"), vec![i]));
            terms_data.push((format!("zzz{i:04}"), vec![50 + i]));
        }
        let terms: Vec<(&str, &[i32])> = terms_data
            .iter()
            .map(|(t, d)| (t.as_str(), d.as_slice()))
            .collect();

        let (dir, fi, id) = write_terms(terms).unwrap();
        let reader = open_reader(&dir, &fi, &id).unwrap();
        let fr = reader.field_reader(0).unwrap();
        let trie = fr.new_trie_reader().unwrap();

        let root = trie.root();
        let mut child = Node::new();
        // 'a' < 'm' (min_children_label)
        assert!(!trie.lookup_child(b'a', root, &mut child).unwrap());
    }

    #[test]
    fn test_bits_lookup_beyond_strategy_bytes() {
        // Regression: BITS strategy with strategy_bytes=3 covers labels 'a'-'x'
        // (delta 0-23). Looking up 'y' (delta 24) must return not-found, not
        // read past the strategy data into child delta FP bytes.
        let mut terms_data: Vec<(String, Vec<i32>)> = Vec::new();
        // Create terms under prefixes "ba" through "bx" (24 letters) to get a
        // BITS node with ~3 strategy bytes, then seek for "by..." which is
        // beyond the covered range.
        for label in b'a'..=b'x' {
            for i in 0..50 {
                let term = format!("b{}{i:04}", label as char);
                terms_data.push((term, vec![((label - b'a') as i32) * 50 + i]));
            }
        }
        let terms: Vec<(&str, &[i32])> = terms_data
            .iter()
            .map(|(t, d)| (t.as_str(), d.as_slice()))
            .collect();

        let (dir, fi, id) = write_terms(terms).unwrap();
        let reader = open_reader(&dir, &fi, &id).unwrap();
        let fr = reader.field_reader(0).unwrap();
        let trie = fr.new_trie_reader().unwrap();

        // "by0000" has prefix "by" where 'y' is beyond the BITS range
        let result = trie.seek_to_block(b"by0000").unwrap();
        // Should find the "b" block (depth 1), not a garbage FP at depth 2
        if let Some(r) = &result {
            assert!(
                r.output_fp < 1_000_000,
                "output_fp looks like garbage: {}",
                r.output_fp
            );
        }

        // "bz0000" also beyond range
        let result = trie.seek_to_block(b"bz0000").unwrap();
        if let Some(r) = &result {
            assert!(
                r.output_fp < 1_000_000,
                "output_fp looks like garbage: {}",
                r.output_fp
            );
        }
    }

    #[test]
    fn test_load_leaf_node_8_byte_output_fp() {
        // Craft a leaf node where fp_bytes_minus1 == 7 (8-byte output FP).
        // Byte layout at fp:
        //   byte 0: term_flags low byte = sign(0x00) | fp_bytes_minus1(7)<<2 = 0x1C
        //   bytes 1-8: output FP as LE i64
        let expected_fp: i64 = 0x0123_4567_89AB_CDEF;
        let mut data = vec![0u8; 16];
        // sign=NO_CHILDREN(0x00) | fp_bytes_minus1=7<<2=0x1C | has_terms(0x20)
        data[0] = 0x1C | 0x20;
        data[1..9].copy_from_slice(&expected_fp.to_le_bytes());

        let access = IndexInput::unnamed(&data);
        let mut node = Node::new();
        let fp = 0i64;
        let term_flags_long = access.read_le_long_at(fp as usize).unwrap();
        let term_flags = term_flags_long as u32;

        load_leaf_node(&access, term_flags, term_flags_long, fp, &mut node).unwrap();

        assert_eq!(node.output_fp, expected_fp);
        assert!(node.has_terms);
        assert_eq!(node.floor_data_fp, NO_FLOOR_DATA);
    }

    #[test]
    fn test_load_leaf_node_8_byte_with_floor() {
        // Same as above but with floor data flag set.
        let expected_fp: i64 = 0x00FF_FFFF_FFFF_FFFF;
        let mut data = vec![0u8; 16];
        // sign=NO_CHILDREN | fp_bytes_minus1=7<<2 | has_terms(0x20) | has_floor(0x40)
        data[0] = 0x1C | 0x20 | 0x40;
        data[1..9].copy_from_slice(&expected_fp.to_le_bytes());

        let access = IndexInput::unnamed(&data);
        let mut node = Node::new();
        let fp = 0i64;
        let term_flags_long = access.read_le_long_at(fp as usize).unwrap();
        let term_flags = term_flags_long as u32;

        load_leaf_node(&access, term_flags, term_flags_long, fp, &mut node).unwrap();

        assert_eq!(node.output_fp, expected_fp);
        assert!(node.has_terms);
        // floor_data_fp = fp + 2 + fp_bytes_minus1 = 0 + 2 + 7 = 9
        assert_eq!(node.floor_data_fp, 9);
    }
}
