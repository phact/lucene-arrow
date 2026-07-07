// SPDX-License-Identifier: Apache-2.0
//! Norms writer (.nvm, .nvd) for per-field normalization values used in scoring.

use std::io;

use log::debug;

use crate::codecs::lucene90::indexed_disi;
use crate::codecs::lucene90::norms_producer::NormsProducer;
use crate::codecs::{codec_footers, codec_headers};
use crate::index::FieldInfo;
use crate::index::index_file_names;
use crate::search::doc_id_set_iterator::NO_MORE_DOCS;
use crate::store::{DataOutput, Directory, IndexOutput};

// File extensions
pub(crate) const DATA_EXTENSION: &str = "nvd";
pub(crate) const META_EXTENSION: &str = "nvm";

// Codec names and versions
pub(crate) const DATA_CODEC: &str = "Lucene90NormsData";
pub(crate) const META_CODEC: &str = "Lucene90NormsMetadata";
pub(crate) const VERSION: i32 = 0;

/// Determines how many bytes are needed per norm value.
fn num_bytes_per_value(min: i64, max: i64) -> u8 {
    if min >= max {
        0
    } else if min >= -128 && max <= 127 {
        1
    } else if min >= -32768 && max <= 32767 {
        2
    } else if min >= i32::MIN as i64 && max <= i32::MAX as i64 {
        4
    } else {
        8
    }
}

/// Writes norms files (.nvm, .nvd) for a segment using a [`NormsProducer`].
///
/// `field_infos` must be sorted by field number.
/// Returns the names of the files written, or an empty vec if no fields provided.
pub(crate) fn write(
    directory: &dyn Directory,
    segment_name: &str,
    segment_suffix: &str,
    segment_id: &[u8; 16],
    field_infos: &[&FieldInfo],
    producer: &dyn NormsProducer,
    max_doc: i32,
) -> io::Result<Vec<String>> {
    if field_infos.is_empty() {
        return Ok(vec![]);
    }

    let nvm_name =
        index_file_names::segment_file_name(segment_name, segment_suffix, META_EXTENSION);
    let nvd_name =
        index_file_names::segment_file_name(segment_name, segment_suffix, DATA_EXTENSION);

    let mut nvm = directory.create_output(&nvm_name)?;
    let mut nvd = directory.create_output(&nvd_name)?;

    codec_headers::write_index_header(&mut *nvm, META_CODEC, VERSION, segment_id, segment_suffix)?;
    codec_headers::write_index_header(&mut *nvd, DATA_CODEC, VERSION, segment_id, segment_suffix)?;

    for &field_info in field_infos {
        add_norms_field(field_info, producer, max_doc, &mut *nvm, &mut *nvd)?;
    }

    // EOF marker
    nvm.write_le_int(-1)?;

    codec_footers::write_footer(&mut *nvm)?;
    codec_footers::write_footer(&mut *nvd)?;

    Ok(vec![nvm_name, nvd_name])
}

/// Writes norms for a single field.
///
/// Three passes over the producer's iterator:
/// 1. Compute num_docs_with_value, min, max
/// 2. Write IndexedDISI bitset (sparse case only)
/// 3. Write norm values (if not constant)
fn add_norms_field(
    field: &FieldInfo,
    norms_producer: &dyn NormsProducer,
    max_doc: i32,
    meta: &mut dyn DataOutput,
    data: &mut dyn IndexOutput,
) -> io::Result<()> {
    // Pass 1: compute count, min, max
    let mut values = match norms_producer.get_norms(field)? {
        Some(v) => v,
        None => {
            // EMPTY pattern: no documents have norms for this field
            debug!(
                "norms: field={:?} (#{}) -> EMPTY pattern",
                field.name(),
                field.number()
            );
            meta.write_le_int(field.number() as i32)?;
            meta.write_le_long(-2)?; // docsWithFieldOffset
            meta.write_le_long(0)?; // docsWithFieldLength
            meta.write_le_short(-1)?; // jumpTableEntryCount
            meta.write_byte(0xFF)?; // denseRankPower
            meta.write_le_int(0)?; // numDocsWithField
            meta.write_byte(0)?; // bytesPerNorm
            meta.write_le_long(0)?; // normsOffset
            return Ok(());
        }
    };
    let mut num_docs_with_value = 0i32;
    let mut min = i64::MAX;
    let mut max = i64::MIN;
    loop {
        let doc = values.next_doc()?;
        if doc == NO_MORE_DOCS {
            break;
        }
        num_docs_with_value += 1;
        let v = values.long_value()?;
        min = min.min(v);
        max = max.max(v);
    }
    // The indexing pipeline may record multiple norms per doc (multi-valued fields),
    // so num_docs_with_value can exceed max_doc. The writer handles this correctly
    // by using the sparse pattern when they differ.

    meta.write_le_int(field.number() as i32)?;

    if num_docs_with_value == 0 {
        debug!(
            "norms: field={:?} (#{}) -> EMPTY pattern",
            field.name(),
            field.number()
        );
        meta.write_le_long(-2)?; // docsWithFieldOffset
        meta.write_le_long(0)?; // docsWithFieldLength
        meta.write_le_short(-1)?; // jumpTableEntryCount
        meta.write_byte(0xFF)?; // denseRankPower
    } else if num_docs_with_value == max_doc {
        debug!(
            "norms: field={:?} (#{}) -> ALL pattern, num_docs_with_field={}",
            field.name(),
            field.number(),
            num_docs_with_value
        );
        meta.write_le_long(-1)?; // docsWithFieldOffset
        meta.write_le_long(0)?; // docsWithFieldLength
        meta.write_le_short(-1)?; // jumpTableEntryCount
        meta.write_byte(0xFF)?; // denseRankPower
    } else {
        // Pass 2: write IndexedDISI bitset (sparse case)
        debug!(
            "norms: field={:?} (#{}) -> SPARSE pattern, num_docs_with_field={}/{}",
            field.name(),
            field.number(),
            num_docs_with_value,
            max_doc
        );
        let offset = data.file_pointer() as i64;
        meta.write_le_long(offset)?; // docsWithFieldOffset
        let mut values = norms_producer.get_norms(field)?.unwrap();
        let mut doc_ids = Vec::with_capacity(num_docs_with_value as usize);
        loop {
            let doc = values.next_doc()?;
            if doc == NO_MORE_DOCS {
                break;
            }
            doc_ids.push(doc);
        }
        let jump_table_entry_count = indexed_disi::write_bit_set(&doc_ids, max_doc, data)?;
        meta.write_le_long(data.file_pointer() as i64 - offset)?; // docsWithFieldLength
        meta.write_le_short(jump_table_entry_count)?;
        meta.write_byte(indexed_disi::DEFAULT_DENSE_RANK_POWER as u8)?;
    }

    meta.write_le_int(num_docs_with_value)?;
    let num_bytes_per_value = num_bytes_per_value(min, max);

    meta.write_byte(num_bytes_per_value)?;
    if num_bytes_per_value == 0 {
        meta.write_le_long(min)?;
    } else {
        meta.write_le_long(data.file_pointer() as i64)?; // normsOffset
        // Pass 3: write norm values
        let mut values = norms_producer.get_norms(field)?.unwrap();
        loop {
            let doc = values.next_doc()?;
            if doc == NO_MORE_DOCS {
                break;
            }
            let v = values.long_value()?;
            write_norm_value(data, v, num_bytes_per_value)?;
        }
    }

    Ok(())
}

/// Writes a single norm value.
fn write_norm_value(out: &mut dyn DataOutput, value: i64, bytes_per_norm: u8) -> io::Result<()> {
    match bytes_per_norm {
        1 => out.write_byte(value as u8),
        2 => out.write_le_short(value as i16),
        4 => out.write_le_int(value as i32),
        8 => out.write_le_long(value),
        _ => unreachable!("invalid bytes_per_norm: {}", bytes_per_norm),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::codecs::codec_footers::FOOTER_LENGTH;
    use crate::codecs::codec_headers::index_header_length;
    use crate::codecs::lucene90::norms_producer::BufferedNormsProducer;
    use crate::document::{DocValuesType, IndexOptions};
    use crate::index::field_infos::PointDimensionConfig;
    use crate::index::pipeline::segment_accumulator::PerFieldNormsData;
    use crate::store::{MemoryDirectory, SharedDirectory};
    use crate::test_util::TestDataReader;
    use assertables::*;

    fn test_directory() -> SharedDirectory {
        MemoryDirectory::create()
    }

    fn make_field_info(name: &str, number: u32) -> FieldInfo {
        FieldInfo::new(
            name.to_string(),
            number,
            false,
            false, // omit_norms = false
            IndexOptions::DocsAndFreqsAndPositions,
            DocValuesType::None,
            PointDimensionConfig::default(),
        )
    }

    fn write_norms(
        dir: &dyn Directory,
        fields: &[(&str, u32, &[i64], &[i32])],
        num_docs: i32,
    ) -> io::Result<Vec<String>> {
        let mut norms_map = HashMap::new();
        let mut field_infos = Vec::new();
        for &(name, number, values, docs) in fields {
            norms_map.insert(
                number,
                PerFieldNormsData {
                    field_name: name.to_string(),
                    docs: docs.to_vec(),
                    values: values.to_vec(),
                },
            );
            field_infos.push(make_field_info(name, number));
        }
        field_infos.sort_by_key(|f| f.number());
        let producer = BufferedNormsProducer::new(&norms_map);
        let field_info_refs: Vec<&FieldInfo> = field_infos.iter().collect();
        write(
            dir,
            "_0",
            "",
            &[0u8; 16],
            &field_info_refs,
            &producer,
            num_docs,
        )
    }

    /// Size of one metadata entry in bytes:
    /// 4 (field_number) + 8 (offset) + 8 (length) + 2 (jump_table) + 1 (rank_power)
    /// + 4 (num_docs) + 1 (bytes_per_norm) + 8 (norms_offset) = 36
    const META_ENTRY_SIZE: usize = 36;

    #[test]
    fn empty_fields_returns_no_files() {
        let dir = test_directory();
        let names = write_norms(&dir, &[], 3).unwrap();
        assert_is_empty!(&names);
    }

    #[test]
    fn all_pattern_1byte_norms() {
        let dir = test_directory();
        let names = write_norms(&dir, &[("contents", 2, &[12, 8, 10], &[0, 1, 2])], 3).unwrap();

        assert_len_eq_x!(&names, 2);
        assert_eq!(names[0], "_0.nvm");
        assert_eq!(names[1], "_0.nvd");

        let nvm = dir.read_file(&names[0]).unwrap();
        let nvd = dir.read_file(&names[1]).unwrap();

        // Both files should start with codec magic
        assert_eq!(&nvm[0..4], &[0x3f, 0xd7, 0x6c, 0x17]);
        assert_eq!(&nvd[0..4], &[0x3f, 0xd7, 0x6c, 0x17]);

        // Parse metadata entry (starts after header)
        let meta_header_len = index_header_length(META_CODEC, "");
        let entry = &nvm[meta_header_len..];

        // field_number = 2
        assert_eq!(&entry[0..4], &2i32.to_le_bytes());

        // docs_with_field_offset = -1 (ALL)
        assert_eq!(&entry[4..12], &(-1i64).to_le_bytes());

        // num_docs_with_field = 3
        assert_eq!(&entry[23..27], &3i32.to_le_bytes());

        // bytes_per_norm = 1
        assert_eq!(entry[27], 1);

        // norms_offset points into .nvd data area
        let data_header_len = index_header_length(DATA_CODEC, "");
        let expected_offset = data_header_len as i64;
        assert_eq!(&entry[28..36], &expected_offset.to_le_bytes());

        // EOF marker = -1
        assert_eq!(
            &nvm[meta_header_len + META_ENTRY_SIZE..meta_header_len + META_ENTRY_SIZE + 4],
            &(-1i32).to_le_bytes()
        );

        // Verify .nvd contains 3 norm bytes
        assert_eq!(nvd[data_header_len], 12u8);
        assert_eq!(nvd[data_header_len + 1], 8u8);
        assert_eq!(nvd[data_header_len + 2], 10u8);

        // Verify footer magic on both files
        let nvm_footer_start = nvm.len() - FOOTER_LENGTH;
        assert_eq!(
            &nvm[nvm_footer_start..nvm_footer_start + 4],
            &[0xc0, 0x28, 0x93, 0xe8]
        );
        let nvd_footer_start = nvd.len() - FOOTER_LENGTH;
        assert_eq!(
            &nvd[nvd_footer_start..nvd_footer_start + 4],
            &[0xc0, 0x28, 0x93, 0xe8]
        );
    }

    #[test]
    fn empty_pattern() {
        let dir = test_directory();
        let names = write_norms(&dir, &[("contents", 0, &[], &[])], 3).unwrap();

        let nvm = dir.read_file(&names[0]).unwrap();
        let meta_header_len = index_header_length(META_CODEC, "");
        let entry = &nvm[meta_header_len..];

        // docs_with_field_offset = -2 (EMPTY)
        assert_eq!(&entry[4..12], &(-2i64).to_le_bytes());

        // num_docs_with_field = 0
        assert_eq!(&entry[23..27], &0i32.to_le_bytes());
    }

    #[test]
    fn constant_norms() {
        let dir = test_directory();
        let names = write_norms(&dir, &[("contents", 1, &[12, 12, 12], &[0, 1, 2])], 3).unwrap();

        let nvm = dir.read_file(&names[0]).unwrap();
        let nvd = dir.read_file(&names[1]).unwrap();
        let meta_header_len = index_header_length(META_CODEC, "");
        let entry = &nvm[meta_header_len..];

        // ALL pattern
        assert_eq!(&entry[4..12], &(-1i64).to_le_bytes());

        // bytes_per_norm = 0 (constant)
        assert_eq!(entry[27], 0);

        // norms_offset = constant value = 12
        assert_eq!(&entry[28..36], &12i64.to_le_bytes());

        // .nvd should have only header + footer (no data bytes)
        let data_header_len = index_header_length(DATA_CODEC, "");
        assert_eq!(nvd.len(), data_header_len + FOOTER_LENGTH);
    }

    #[test]
    fn multiple_fields() {
        let dir = test_directory();
        let names = write_norms(
            &dir,
            &[
                ("alpha", 0, &[5, 5, 5], &[0, 1, 2]),
                ("beta", 1, &[10, 20, 30], &[0, 1, 2]),
            ],
            3,
        )
        .unwrap();

        let nvm = dir.read_file(&names[0]).unwrap();
        let nvd = dir.read_file(&names[1]).unwrap();
        let meta_header_len = index_header_length(META_CODEC, "");

        // First entry: "alpha" (field 0), constant norms
        let entry0 = &nvm[meta_header_len..];
        assert_eq!(&entry0[0..4], &0i32.to_le_bytes());
        assert_eq!(entry0[27], 0); // bytes_per_norm = 0 (constant)
        assert_eq!(&entry0[28..36], &5i64.to_le_bytes());

        // Second entry: "beta" (field 1), 1-byte norms
        let entry1 = &nvm[meta_header_len + META_ENTRY_SIZE..];
        assert_eq!(&entry1[0..4], &1i32.to_le_bytes());
        assert_eq!(entry1[27], 1);

        // "beta" norm values in .nvd
        let data_header_len = index_header_length(DATA_CODEC, "");
        assert_eq!(nvd[data_header_len], 10u8);
        assert_eq!(nvd[data_header_len + 1], 20u8);
        assert_eq!(nvd[data_header_len + 2], 30u8);

        // EOF marker after second entry
        assert_eq!(
            &nvm[meta_header_len + 2 * META_ENTRY_SIZE..meta_header_len + 2 * META_ENTRY_SIZE + 4],
            &(-1i32).to_le_bytes()
        );
    }

    #[test]
    fn sparse_norms() {
        let dir = test_directory();
        // 2 docs with norms out of 5 total
        let names = write_norms(&dir, &[("contents", 0, &[12, 8], &[1, 3])], 5).unwrap();

        let nvm = dir.read_file(&names[0]).unwrap();
        let nvd = dir.read_file(&names[1]).unwrap();
        let meta_header_len = index_header_length(META_CODEC, "");
        let mut reader = TestDataReader::new(&nvm[meta_header_len..], 0);

        // field_number = 0
        assert_eq!(reader.read_le_int(), 0);

        // SPARSE: offset >= 0
        let docs_with_field_offset = reader.read_le_long();
        assert_ge!(docs_with_field_offset, 0);

        // docsWithFieldLength > 0
        let docs_with_field_length = reader.read_le_long();
        assert_gt!(docs_with_field_length, 0);

        let _jump_table_entry_count = reader.read_le_short();
        assert_eq!(reader.read_byte(), 9); // denseRankPower

        // numDocsWithField = 2
        assert_eq!(reader.read_le_int(), 2);

        // bytesPerNorm = 1
        assert_eq!(reader.read_byte(), 1);

        // normsOffset points after IndexedDISI data
        let norms_offset = reader.read_le_long();
        let disi_end = docs_with_field_offset + docs_with_field_length;
        assert_eq!(norms_offset, disi_end);

        // Verify norm values in .nvd
        assert_eq!(nvd[norms_offset as usize], 12u8);
        assert_eq!(nvd[norms_offset as usize + 1], 8u8);
    }

    #[test]
    fn sparse_constant_norms() {
        let dir = test_directory();
        // 3 docs with identical norms out of 5 total
        let names = write_norms(&dir, &[("title", 0, &[42, 42, 42], &[0, 2, 4])], 5).unwrap();

        let nvm = dir.read_file(&names[0]).unwrap();
        let meta_header_len = index_header_length(META_CODEC, "");
        let mut reader = TestDataReader::new(&nvm[meta_header_len..], 0);

        assert_eq!(reader.read_le_int(), 0); // field_number

        // SPARSE: offset >= 0
        assert_ge!(reader.read_le_long(), 0);

        let _docs_with_field_length = reader.read_le_long();
        let _jump_table_entry_count = reader.read_le_short();
        assert_eq!(reader.read_byte(), 9); // denseRankPower

        assert_eq!(reader.read_le_int(), 3); // numDocsWithField
        assert_eq!(reader.read_byte(), 0); // bytesPerNorm = 0 (constant)
        assert_eq!(reader.read_le_long(), 42); // constant value
    }

    #[test]
    fn num_bytes_per_value_ranges() {
        // Constant
        assert_eq!(num_bytes_per_value(5, 5), 0);

        // 1 byte
        assert_eq!(num_bytes_per_value(-128, 127), 1);

        // 2 bytes
        assert_eq!(num_bytes_per_value(0, 128), 2);
        assert_eq!(num_bytes_per_value(-32768, 32767), 2);

        // 4 bytes
        assert_eq!(num_bytes_per_value(0, 32768), 4);

        // 8 bytes
        assert_eq!(num_bytes_per_value(i32::MIN as i64 - 1, 0), 8);
    }
}
