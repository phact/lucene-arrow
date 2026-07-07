// SPDX-License-Identifier: Apache-2.0

//! Field metadata types.
//!
//! [`FieldInfo`] describes a single field's schema (index options, doc values,
//! point dimensions, etc.) and [`FieldInfos`] is an immutable collection
//! indexed by both name and field number.

use std::collections::HashMap;

use mem_dbg::MemSize;

use crate::document::{DocValuesType, IndexOptions};

/// Configuration for point (BKD tree) dimensions.
#[derive(Clone, Copy, Debug, Default, MemSize)]
#[mem_size(flat)]
pub struct PointDimensionConfig {
    pub dimension_count: u32,
    pub index_dimension_count: u32,
    pub num_bytes: u32,
}

impl PointDimensionConfig {
    pub fn has_values(&self) -> bool {
        self.dimension_count > 0
    }
}

/// Information about a single field in the index.
#[derive(Clone, Debug, MemSize)]
pub struct FieldInfo {
    name: String,
    number: u32,
    store_term_vector: bool,
    omit_norms: bool,
    pub(crate) store_payloads: bool,
    index_options: IndexOptions,
    doc_values_type: DocValuesType,
    /// Doc values skip index type: 0 = NONE, 1 = RANGE.
    pub(crate) doc_values_skip_index_type: u8,
    pub(crate) dv_gen: i64,
    attributes: HashMap<String, String>,
    point_config: PointDimensionConfig,
    vector_dimension: u32,
    pub(crate) soft_deletes_field: bool,
    pub(crate) is_parent_field: bool,
}

impl FieldInfo {
    pub fn new(
        name: String,
        number: u32,
        store_term_vector: bool,
        omit_norms: bool,
        index_options: IndexOptions,
        doc_values_type: DocValuesType,
        point_config: PointDimensionConfig,
    ) -> Self {
        Self {
            name,
            number,
            store_term_vector,
            omit_norms,
            store_payloads: false,
            index_options,
            doc_values_type,
            doc_values_skip_index_type: 0,
            dv_gen: -1,
            attributes: HashMap::new(),
            point_config,
            vector_dimension: 0,
            soft_deletes_field: false,
            is_parent_field: false,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn number(&self) -> u32 {
        self.number
    }

    pub fn store_term_vector(&self) -> bool {
        self.store_term_vector
    }

    pub fn omit_norms(&self) -> bool {
        self.omit_norms
    }

    pub fn store_payloads(&self) -> bool {
        self.store_payloads
    }

    pub fn index_options(&self) -> IndexOptions {
        self.index_options
    }

    pub fn doc_values_type(&self) -> DocValuesType {
        self.doc_values_type
    }

    pub fn dv_gen(&self) -> i64 {
        self.dv_gen
    }

    pub fn attributes(&self) -> &HashMap<String, String> {
        &self.attributes
    }

    pub fn point_config(&self) -> PointDimensionConfig {
        self.point_config
    }

    pub fn vector_dimension(&self) -> u32 {
        self.vector_dimension
    }

    pub fn soft_deletes_field(&self) -> bool {
        self.soft_deletes_field
    }

    pub fn is_parent_field(&self) -> bool {
        self.is_parent_field
    }

    pub fn is_indexed(&self) -> bool {
        self.index_options != IndexOptions::None
    }

    /// Returns true if this field is indexed and has norms enabled.
    pub fn has_norms(&self) -> bool {
        self.is_indexed() && !self.omit_norms
    }

    pub fn has_payloads(&self) -> bool {
        self.store_payloads
    }

    pub fn has_vectors(&self) -> bool {
        self.store_term_vector
    }

    pub fn has_doc_values(&self) -> bool {
        self.doc_values_type != DocValuesType::None
    }

    pub fn has_point_values(&self) -> bool {
        self.point_config.has_values()
    }

    pub fn has_vector_values(&self) -> bool {
        self.vector_dimension > 0
    }

    /// Returns the attribute value for the given key, if any.
    pub fn get_attribute(&self, key: &str) -> Option<&str> {
        self.attributes.get(key).map(|s| s.as_str())
    }

    /// Sets an attribute key-value pair.
    pub fn put_attribute(&mut self, key: String, value: String) {
        self.attributes.insert(key, value);
    }
}

/// A collection of FieldInfo objects indexed by name and number.
#[derive(Clone, Debug)]
pub struct FieldInfos {
    fields: Box<[FieldInfo]>,
    by_name: HashMap<String, usize>,
    by_number: HashMap<u32, usize>,
    has_freq: bool,
    has_postings: bool,
    has_prox: bool,
    has_payloads: bool,
    has_offsets: bool,
    has_vectors: bool,
    has_norms: bool,
    has_doc_values: bool,
    has_point_values: bool,
    has_vector_values: bool,
}

impl FieldInfos {
    /// Creates a new `FieldInfos` and computes aggregate flags from the field list.
    pub fn new(fields: Vec<FieldInfo>) -> Self {
        let mut by_name = HashMap::new();
        let mut by_number = HashMap::new();
        let mut has_freq = false;
        let mut has_postings = false;
        let mut has_prox = false;
        let mut has_payloads = false;
        let mut has_offsets = false;
        let mut has_vectors = false;
        let mut has_norms = false;
        let mut has_doc_values = false;
        let mut has_point_values = false;
        let mut has_vector_values = false;

        for (idx, fi) in fields.iter().enumerate() {
            by_name.insert(fi.name().to_string(), idx);
            by_number.insert(fi.number(), idx);

            if fi.index_options() != IndexOptions::None {
                has_postings = true;
            }
            if fi.index_options() >= IndexOptions::DocsAndFreqs {
                has_freq = true;
            }
            if fi.index_options() >= IndexOptions::DocsAndFreqsAndPositions {
                has_prox = true;
            }
            if fi.index_options() >= IndexOptions::DocsAndFreqsAndPositionsAndOffsets {
                has_offsets = true;
            }
            if fi.store_payloads() {
                has_payloads = true;
            }
            if fi.store_term_vector() {
                has_vectors = true;
            }
            if fi.has_norms() {
                has_norms = true;
            }
            if fi.doc_values_type() != DocValuesType::None {
                has_doc_values = true;
            }
            if fi.point_config().has_values() {
                has_point_values = true;
            }
            if fi.vector_dimension() > 0 {
                has_vector_values = true;
            }
        }

        Self {
            fields: fields.into_boxed_slice(),
            by_name,
            by_number,
            has_freq,
            has_postings,
            has_prox,
            has_payloads,
            has_offsets,
            has_vectors,
            has_norms,
            has_doc_values,
            has_point_values,
            has_vector_values,
        }
    }

    pub fn field_info_by_name(&self, name: &str) -> Option<&FieldInfo> {
        self.by_name.get(name).map(|&idx| &self.fields[idx])
    }

    pub fn field_info_by_number(&self, number: u32) -> Option<&FieldInfo> {
        self.by_number.get(&number).map(|&idx| &self.fields[idx])
    }

    pub fn iter(&self) -> impl Iterator<Item = &FieldInfo> {
        self.fields.iter()
    }

    pub fn len(&self) -> usize {
        self.fields.len()
    }

    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    pub fn has_freq(&self) -> bool {
        self.has_freq
    }

    pub fn has_postings(&self) -> bool {
        self.has_postings
    }

    pub fn has_prox(&self) -> bool {
        self.has_prox
    }

    pub fn has_payloads(&self) -> bool {
        self.has_payloads
    }

    pub fn has_offsets(&self) -> bool {
        self.has_offsets
    }

    pub fn has_vectors(&self) -> bool {
        self.has_vectors
    }

    pub fn has_norms(&self) -> bool {
        self.has_norms
    }

    pub fn has_doc_values(&self) -> bool {
        self.has_doc_values
    }

    pub fn has_point_values(&self) -> bool {
        self.has_point_values
    }

    pub fn has_vector_values(&self) -> bool {
        self.has_vector_values
    }
}

#[cfg(test)]
mod tests {
    use assertables::assert_len_eq_x;
    use assertables::assert_none;
    use assertables::assert_some;

    use super::*;
    use crate::test_util;

    fn make_test_field_info(name: &str, number: u32, index_opts: IndexOptions) -> FieldInfo {
        test_util::make_field_info(name, number, false, index_opts, DocValuesType::None)
    }

    #[test]
    fn test_field_info_basics() {
        let fi = make_test_field_info("contents", 0, IndexOptions::DocsAndFreqsAndPositions);
        assert!(fi.is_indexed());
        assert!(fi.has_norms());
        assert!(!fi.has_payloads());
        assert!(!fi.has_doc_values());
        assert!(!fi.has_point_values());
    }

    #[test]
    fn test_field_info_keyword() {
        let fi = FieldInfo::new(
            "path".to_string(),
            0,
            false,
            true, // omitNorms
            IndexOptions::Docs,
            DocValuesType::SortedSet,
            PointDimensionConfig::default(),
        );
        assert!(fi.is_indexed());
        assert!(!fi.has_norms()); // omitNorms=true
        assert!(fi.has_doc_values());
    }

    #[test]
    fn test_field_infos_aggregation() {
        let fi_path = FieldInfo::new(
            "path".to_string(),
            0,
            false,
            true,
            IndexOptions::Docs,
            DocValuesType::SortedSet,
            PointDimensionConfig::default(),
        );
        let fi_modified = FieldInfo::new(
            "modified".to_string(),
            1,
            false,
            false,
            IndexOptions::None,
            DocValuesType::SortedNumeric,
            PointDimensionConfig {
                dimension_count: 1,
                index_dimension_count: 1,
                num_bytes: 8,
            },
        );
        let fi_contents = FieldInfo::new(
            "contents".to_string(),
            2,
            false,
            false,
            IndexOptions::DocsAndFreqsAndPositions,
            DocValuesType::None,
            PointDimensionConfig::default(),
        );

        let fis = FieldInfos::new(vec![fi_path, fi_modified, fi_contents]);
        assert_len_eq_x!(&fis, 3);
        assert!(fis.has_postings());
        assert!(fis.has_freq());
        assert!(fis.has_prox());
        assert!(!fis.has_offsets());
        assert!(!fis.has_payloads());
        assert!(!fis.has_vectors());
        assert!(fis.has_norms());
        assert!(fis.has_doc_values());
        assert!(fis.has_point_values());
    }

    #[test]
    fn test_field_infos_lookup() {
        let fi = make_test_field_info("test", 42, IndexOptions::Docs);
        let fis = FieldInfos::new(vec![fi]);

        assert_some!(fis.field_info_by_name("test"));
        assert_eq!(fis.field_info_by_name("test").unwrap().number(), 42);
        assert_some!(fis.field_info_by_number(42));
        assert_none!(fis.field_info_by_name("missing"));
    }
}
