// SPDX-License-Identifier: Apache-2.0

//! Points/BKD reader for the Lucene90 points format.
//!
//! Reads `.kdm` (metadata), `.kdi` (index), and `.kdd` (data) files written by
//! `super::points::write()`. Metadata is read eagerly from `.kdm`; tree and
//! leaf data in `.kdi`/`.kdd` are available via retained file handles.

use std::fmt;
use std::io;

use log::debug;

use crate::codecs::codec_file_handle::{CodecFileHandle, IndexFile};
use crate::codecs::codec_headers;
use crate::codecs::codec_headers::check_header;
use crate::codecs::lucene90::points::{BKD_CODEC, BKD_VERSION, PointsFieldData};
use crate::index::{FieldInfo, FieldInfos};
use crate::store::{Directory, FileBacking, IndexInput};

/// Per-field BKD tree metadata read eagerly from `.kdm`.
#[derive(Clone)]
struct BkdEntry {
    num_leaves: u32,
    point_count: i64,
    doc_count: i32,
}

/// Reads points/BKD metadata for a segment.
///
/// Constructor matches the lifecycle of Java's `Lucene90PointsReader`:
/// opens `.kdi` and `.kdd` first (keeps backings), then reads `.kdm` metadata,
/// then validates file lengths.
pub struct PointsReader {
    /// Per-field BKD metadata indexed by field number. `None` for fields without points.
    entries: Box<[Option<BkdEntry>]>,
    /// Owned backing for the `.kdi` index file.
    #[expect(dead_code)]
    index_in: FileBacking,
    /// Owned backing for the `.kdd` data file.
    #[expect(dead_code)]
    data_in: FileBacking,
}

impl PointsReader {
    /// Opens points files (`.kdi`, `.kdd`, `.kdm`) for the given segment.
    ///
    /// 1. Open and validate `.kdi` (index) — keep backing
    /// 2. Open and validate `.kdd` (data) — keep backing
    /// 3. Open `.kdm` (meta), verify full CRC, read all per-field entries
    /// 4. Validate `.kdi` and `.kdd` file lengths via `retrieve_checksum_with_length`
    pub fn open(
        directory: &dyn Directory,
        segment_name: &str,
        segment_suffix: &str,
        segment_id: &[u8; codec_headers::ID_LENGTH],
        field_infos: &FieldInfos,
    ) -> io::Result<Self> {
        let index_handle = CodecFileHandle::open(
            directory,
            IndexFile::PointsIndex,
            segment_name,
            segment_id,
            segment_suffix,
        )?;
        let data_handle = CodecFileHandle::open(
            directory,
            IndexFile::PointsData,
            segment_name,
            segment_id,
            segment_suffix,
        )?;
        let meta_handle = CodecFileHandle::open(
            directory,
            IndexFile::PointsMeta,
            segment_name,
            segment_id,
            segment_suffix,
        )?;

        let mut meta = meta_handle.body();
        let entries = read_fields(&mut meta, field_infos)?;

        let index_length = meta.read_le_long()?;
        let data_length = meta.read_le_long()?;

        // Validate sibling file lengths against what .kdm claimed.
        index_handle.verify_length(index_length)?;
        data_handle.verify_length(data_length)?;

        debug!(
            "points_reader: opened {} entries for segment {segment_name}",
            entries.iter().filter(|e| e.is_some()).count()
        );

        Ok(Self {
            entries,
            index_in: index_handle.into_backing(),
            data_in: data_handle.into_backing(),
        })
    }

    /// Returns the total number of indexed points for a field.
    pub fn point_count(&self, field_number: u32) -> Option<i64> {
        self.entry(field_number).map(|e| e.point_count)
    }

    /// Returns the number of documents with points for a field.
    pub fn doc_count(&self, field_number: u32) -> Option<i32> {
        self.entry(field_number).map(|e| e.doc_count)
    }

    /// Returns the number of leaf nodes in the BKD tree for a field.
    pub fn num_leaves(&self, field_number: u32) -> Option<u32> {
        self.entry(field_number).map(|e| e.num_leaves)
    }

    fn entry(&self, field_number: u32) -> Option<&BkdEntry> {
        self.entries
            .get(field_number as usize)
            .and_then(|opt| opt.as_ref())
    }
}

// ---------------------------------------------------------------------------
// PointValues and PointsProducer traits
// ---------------------------------------------------------------------------

/// Per-field access to point values and dimensional metadata.
pub trait PointValues {
    /// Number of dimensions per point.
    fn num_dimensions(&self) -> u32;

    /// Number of index dimensions (may be less than `num_dimensions` for range fields).
    fn num_index_dimensions(&self) -> u32;

    /// Bytes per dimension value.
    fn bytes_per_dimension(&self) -> u32;

    /// Total number of indexed points.
    fn size(&self) -> usize;

    /// Returns all `(doc_id, packed_value)` pairs for this field.
    fn points(&self) -> &[(i32, Vec<u8>)];
}

/// Produces per-field point values.
///
/// Both file-backed readers and in-memory buffered producers implement this trait.
pub trait PointsProducer: fmt::Debug {
    /// Returns a [`PointValues`] for the given field, or `None` if absent.
    fn get_points(&self, field_info: &FieldInfo) -> io::Result<Option<Box<dyn PointValues + '_>>>;
}

// ---------------------------------------------------------------------------
// BufferedPointsProducer — in-memory points from the indexing pipeline
// ---------------------------------------------------------------------------

/// Per-field point data stored in memory.
#[derive(Debug)]
struct BufferedFieldPoints {
    dimension_count: u32,
    index_dimension_count: u32,
    bytes_per_dim: u32,
    points: Vec<(i32, Vec<u8>)>,
}

/// In-memory [`PointsProducer`] wrapping indexing pipeline buffers.
///
/// Each call to [`get_points`](PointsProducer::get_points) returns a
/// [`PointValues`] that borrows from the buffered data.
#[derive(Debug)]
pub struct BufferedPointsProducer {
    fields: Vec<Option<BufferedFieldPoints>>,
}

impl BufferedPointsProducer {
    /// Creates a new buffered producer from point field data.
    pub fn new(fields_data: &[PointsFieldData]) -> Self {
        let max_field = fields_data
            .iter()
            .map(|f| f.field_number as usize + 1)
            .max()
            .unwrap_or(0);
        let mut fields = Vec::with_capacity(max_field);
        fields.resize_with(max_field, || None);

        for f in fields_data {
            fields[f.field_number as usize] = Some(BufferedFieldPoints {
                dimension_count: f.dimension_count,
                index_dimension_count: f.index_dimension_count,
                bytes_per_dim: f.bytes_per_dim,
                points: f.points.clone(),
            });
        }

        Self { fields }
    }
}

impl PointsProducer for BufferedPointsProducer {
    fn get_points(&self, field_info: &FieldInfo) -> io::Result<Option<Box<dyn PointValues + '_>>> {
        let field_data = match self.fields.get(field_info.number() as usize) {
            Some(Some(data)) => data,
            _ => return Ok(None),
        };

        if field_data.points.is_empty() {
            return Ok(None);
        }

        Ok(Some(Box::new(BufferedPointValues {
            dimension_count: field_data.dimension_count,
            index_dimension_count: field_data.index_dimension_count,
            bytes_per_dim: field_data.bytes_per_dim,
            points: &field_data.points,
        })))
    }
}

/// [`PointValues`] borrowing from in-memory buffered data.
struct BufferedPointValues<'a> {
    dimension_count: u32,
    index_dimension_count: u32,
    bytes_per_dim: u32,
    points: &'a [(i32, Vec<u8>)],
}

impl PointValues for BufferedPointValues<'_> {
    fn num_dimensions(&self) -> u32 {
        self.dimension_count
    }

    fn num_index_dimensions(&self) -> u32 {
        self.index_dimension_count
    }

    fn bytes_per_dimension(&self) -> u32 {
        self.bytes_per_dim
    }

    fn size(&self) -> usize {
        self.points.len()
    }

    fn points(&self) -> &[(i32, Vec<u8>)] {
        self.points
    }
}

// ---------------------------------------------------------------------------
// PointsProducer impl for PointsReader (stub)
// ---------------------------------------------------------------------------

impl fmt::Debug for PointsReader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PointsReader")
            .field("entries", &self.entries.len())
            .finish()
    }
}

impl PointsProducer for PointsReader {
    fn get_points(&self, _field_info: &FieldInfo) -> io::Result<Option<Box<dyn PointValues + '_>>> {
        todo!("disk-backed point value reading for merge path")
    }
}

// ---------------------------------------------------------------------------
// .kdm metadata reading
// ---------------------------------------------------------------------------

/// Reads all per-field BKD metadata entries from `.kdm`.
fn read_fields(
    meta: &mut IndexInput<'_>,
    field_infos: &FieldInfos,
) -> io::Result<Box<[Option<BkdEntry>]>> {
    let mut entries: Vec<Option<BkdEntry>> = vec![None; field_infos.len()];

    loop {
        let field_number = meta.read_le_int()?;
        if field_number == -1 {
            break;
        }
        if field_number < 0 {
            return Err(io::Error::other(format!(
                "Illegal field number: {field_number}"
            )));
        }

        let field_number = field_number as u32;
        let _info = field_infos
            .field_info_by_number(field_number)
            .ok_or_else(|| io::Error::other(format!("invalid field number: {field_number}")))?;

        let entry = read_bkd_entry(meta)?;
        entries[field_number as usize] = Some(entry);
    }

    Ok(entries.into_boxed_slice())
}

/// Reads a single BKD metadata entry (one per point field).
fn read_bkd_entry(meta: &mut IndexInput<'_>) -> io::Result<BkdEntry> {
    check_header(meta, BKD_CODEC, BKD_VERSION, BKD_VERSION)?;

    let _num_dims = meta.read_vint()? as u32;
    let num_index_dims = meta.read_vint()? as u32;
    let _max_points_in_leaf = meta.read_vint()?;
    let bytes_per_dim = meta.read_vint()? as u32;
    let num_leaves = meta.read_vint()? as u32;

    // Skip min/max packed values (numIndexDims × bytesPerDim bytes each)
    let packed_len = (num_index_dims * bytes_per_dim) as usize;
    meta.skip_bytes(packed_len * 2)?;

    let point_count = meta.read_vlong()?;
    let doc_count = meta.read_vint()?;

    // Skip file pointers (needed for future tree reads)
    let _num_index_bytes = meta.read_vint()?;
    let _data_start_fp = meta.read_le_long()?;
    let _index_start_fp = meta.read_le_long()?;

    Ok(BkdEntry {
        num_leaves,
        point_count,
        doc_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codecs::lucene90::points::{self, PointsFieldData};
    use crate::document::{DocValuesType, IndexOptions};
    use crate::index::{FieldInfo, FieldInfos, PointDimensionConfig};
    use crate::store::{MemoryDirectory, SharedDirectory};
    use assertables::*;

    fn test_directory() -> SharedDirectory {
        MemoryDirectory::create()
    }

    fn make_point_field(
        name: &str,
        number: u32,
        dims: u32,
        index_dims: u32,
        num_bytes: u32,
    ) -> FieldInfo {
        FieldInfo::new(
            name.to_string(),
            number,
            false,
            true,
            IndexOptions::None,
            DocValuesType::None,
            PointDimensionConfig {
                dimension_count: dims,
                index_dimension_count: index_dims,
                num_bytes,
            },
        )
    }

    fn make_points_field(
        name: &str,
        number: u32,
        dims: u32,
        index_dims: u32,
        num_bytes: u32,
        values: Vec<(i32, Vec<u8>)>,
    ) -> PointsFieldData {
        PointsFieldData {
            field_name: name.to_string(),
            field_number: number,
            dimension_count: dims,
            index_dimension_count: index_dims,
            bytes_per_dim: num_bytes,
            points: values,
        }
    }

    fn write_and_read(field_infos: &FieldInfos, fields: &[PointsFieldData]) -> PointsReader {
        let segment_id = [0u8; 16];
        let dir = test_directory();
        let producer = BufferedPointsProducer::new(fields);
        let fi_refs: Vec<&FieldInfo> = field_infos.iter().collect();
        points::write(&dir, "_0", "", &segment_id, &fi_refs, &producer).unwrap();
        PointsReader::open(&dir, "_0", "", &segment_id, field_infos).unwrap()
    }

    #[test]
    fn test_1d_int_field() {
        let fi = make_point_field("size", 0, 1, 1, 4);
        let field_infos = FieldInfos::new(vec![fi]);

        let fields = vec![make_points_field(
            "size",
            0,
            1,
            1,
            4,
            vec![
                (0, 100i32.to_be_bytes().to_vec()),
                (1, 200i32.to_be_bytes().to_vec()),
                (2, 300i32.to_be_bytes().to_vec()),
            ],
        )];

        let reader = write_and_read(&field_infos, &fields);
        assert_some_eq_x!(reader.point_count(0), 3);
        assert_some_eq_x!(reader.doc_count(0), 3);
    }

    #[test]
    fn test_2d_latlon_field() {
        let fi = make_point_field("location", 0, 2, 2, 4);
        let field_infos = FieldInfos::new(vec![fi]);

        let mut point1 = Vec::new();
        point1.extend_from_slice(&10i32.to_be_bytes());
        point1.extend_from_slice(&20i32.to_be_bytes());
        let mut point2 = Vec::new();
        point2.extend_from_slice(&30i32.to_be_bytes());
        point2.extend_from_slice(&40i32.to_be_bytes());

        let fields = vec![make_points_field(
            "location",
            0,
            2,
            2,
            4,
            vec![(0, point1), (1, point2)],
        )];

        let reader = write_and_read(&field_infos, &fields);
        assert_some_eq_x!(reader.point_count(0), 2);
        assert_some_eq_x!(reader.doc_count(0), 2);
        assert_some_eq_x!(reader.num_leaves(0), 1);
    }

    #[test]
    fn test_multiple_fields() {
        let fi_size = make_point_field("size", 0, 1, 1, 4);
        let fi_loc = make_point_field("location", 1, 2, 2, 4);
        let field_infos = FieldInfos::new(vec![fi_size, fi_loc]);

        let mut loc_point = Vec::new();
        loc_point.extend_from_slice(&10i32.to_be_bytes());
        loc_point.extend_from_slice(&20i32.to_be_bytes());

        let fields = vec![
            make_points_field(
                "size",
                0,
                1,
                1,
                4,
                vec![
                    (0, 100i32.to_be_bytes().to_vec()),
                    (1, 200i32.to_be_bytes().to_vec()),
                ],
            ),
            make_points_field("location", 1, 2, 2, 4, vec![(0, loc_point)]),
        ];

        let reader = write_and_read(&field_infos, &fields);
        assert_some_eq_x!(reader.point_count(0), 2);
        assert_some_eq_x!(reader.doc_count(0), 2);
        assert_some_eq_x!(reader.point_count(1), 1);
        assert_some_eq_x!(reader.doc_count(1), 1);
    }

    #[test]
    fn test_nonexistent_field() {
        let fi = make_point_field("size", 0, 1, 1, 4);
        let field_infos = FieldInfos::new(vec![fi]);

        let fields = vec![make_points_field(
            "size",
            0,
            1,
            1,
            4,
            vec![(0, 42i32.to_be_bytes().to_vec())],
        )];

        let reader = write_and_read(&field_infos, &fields);
        assert_none!(reader.point_count(99));
        assert_none!(reader.doc_count(99));
    }

    #[test]
    fn test_8byte_long_field() {
        let fi = make_point_field("modified", 0, 1, 1, 8);
        let field_infos = FieldInfos::new(vec![fi]);

        let fields = vec![make_points_field(
            "modified",
            0,
            1,
            1,
            8,
            vec![
                (0, 1000i64.to_be_bytes().to_vec()),
                (1, 2000i64.to_be_bytes().to_vec()),
            ],
        )];

        let reader = write_and_read(&field_infos, &fields);
        assert_some_eq_x!(reader.point_count(0), 2);
        assert_some_eq_x!(reader.doc_count(0), 2);
        assert_some_eq_x!(reader.num_leaves(0), 1);
    }

    #[test]
    fn test_open_reads_files_from_indexer_pipeline() {
        // End-to-end check: drive the real indexing pipeline (PointsConsumer)
        // to produce .kdi/.kdd/.kdm files, then open them through the migrated
        // PointsReader::open path. Exercises Directory::open_file +
        // FileBacking + IndexInput together against bytes produced by
        // the indexer, not hand-built fixtures.
        use crate::index::field::int_field;
        use crate::index::pipeline::consumer::FieldConsumer;
        use crate::index::pipeline::points_consumer::PointsConsumer;
        use crate::index::pipeline::segment_accumulator::SegmentAccumulator;
        use crate::index::pipeline::segment_context::SegmentContext;

        let context = SegmentContext {
            directory: MemoryDirectory::create(),
            segment_name: "_0".to_string(),
            segment_id: [0u8; 16],
        };
        let mut consumer = PointsConsumer::new();
        let mut acc = SegmentAccumulator::new();

        for doc_id in 0..3 {
            let field = int_field("count").value(doc_id * 10);
            consumer.start_document(doc_id).unwrap();
            consumer.start_field(0, &field, &mut acc).unwrap();
            consumer.finish_field(0, &field, &mut acc).unwrap();
            consumer
                .finish_document(doc_id, &mut acc, &context)
                .unwrap();
        }

        let names = consumer.flush(&context, &acc).unwrap();
        assert_len_eq_x!(&names, 3);

        let fi = make_point_field("count", 0, 1, 1, 4);
        let field_infos = FieldInfos::new(vec![fi]);
        let reader = PointsReader::open(
            &*context.directory,
            &context.segment_name,
            "",
            &context.segment_id,
            &field_infos,
        )
        .unwrap();

        assert_some_eq_x!(reader.point_count(0), 3);
        assert_some_eq_x!(reader.doc_count(0), 3);
        assert_some_eq_x!(reader.num_leaves(0), 1);
    }

    #[test]
    fn test_truncated_data_file_detected() {
        let fi = make_point_field("size", 0, 1, 1, 4);
        let field_infos = FieldInfos::new(vec![fi]);

        let segment_id = [0u8; 16];
        let dir = test_directory();
        let fields = vec![make_points_field(
            "size",
            0,
            1,
            1,
            4,
            vec![(0, 42i32.to_be_bytes().to_vec())],
        )];
        let producer = BufferedPointsProducer::new(&fields);
        let fi_refs: Vec<&FieldInfo> = field_infos.iter().collect();
        points::write(&dir, "_0", "", &segment_id, &fi_refs, &producer).unwrap();

        // Truncate the .kdd file
        let mem_dir = MemoryDirectory::create();
        for name in dir.list_all().unwrap() {
            let data = dir.read_file(&name).unwrap();
            if name.ends_with(".kdd") {
                mem_dir.write_file(&name, &data[..data.len() - 4]).unwrap();
            } else {
                mem_dir.write_file(&name, &data).unwrap();
            }
        }

        let result = PointsReader::open(&mem_dir, "_0", "", &segment_id, &field_infos);
        assert!(result.is_err(), "should detect truncated .kdd");
    }
}
