// SPDX-License-Identifier: Apache-2.0

//! Geographic coordinate quantization.
//!
//! Encodes latitude/longitude as 32-bit integers for use in point fields.

const LAT_SCALE: f64 = (1u64 << 32) as f64 / 180.0;
const LAT_DECODE: f64 = 1.0 / LAT_SCALE;
const LON_SCALE: f64 = (1u64 << 32) as f64 / 360.0;
const LON_DECODE: f64 = 1.0 / LON_SCALE;

/// Quantizes latitude (-90..90) to a 32-bit integer (rounding toward -90).
///
/// Exact +90.0 is handled as a special case to avoid overflow.
pub fn encode_latitude(lat: f64) -> i32 {
    assert!(
        (-90.0..=90.0).contains(&lat),
        "latitude {lat} out of range [-90, 90]"
    );
    let lat = if lat == 90.0 {
        f64::from_bits(90.0f64.to_bits() - 1)
    } else {
        lat
    };
    (lat / LAT_DECODE).floor() as i32
}

/// Quantizes longitude (-180..180) to a 32-bit integer (rounding toward -180).
///
/// Exact +180.0 is handled as a special case to avoid overflow.
pub fn encode_longitude(lon: f64) -> i32 {
    assert!(
        (-180.0..=180.0).contains(&lon),
        "longitude {lon} out of range [-180, 180]"
    );
    let lon = if lon == 180.0 {
        f64::from_bits(180.0f64.to_bits() - 1)
    } else {
        lon
    };
    (lon / LON_DECODE).floor() as i32
}

/// Decodes a quantized latitude back to a double.
#[cfg(test)]
pub fn decode_latitude(encoded: i32) -> f64 {
    encoded as f64 * LAT_DECODE
}

/// Decodes a quantized longitude back to a double.
#[cfg(test)]
pub fn decode_longitude(encoded: i32) -> f64 {
    encoded as f64 * LON_DECODE
}

#[cfg(test)]
mod tests {
    use super::*;

    // Ported from org.apache.lucene.geo.TestGeoEncodingUtils

    #[test]
    fn test_encode_latitude_roundtrip() {
        for &lat in &[0.0, 45.0, -45.0, -90.0, 89.999] {
            let encoded = encode_latitude(lat);
            let decoded = decode_latitude(encoded);
            assert_in_delta!(decoded, lat, 4.2e-8);
        }
    }

    #[test]
    fn test_encode_longitude_roundtrip() {
        for &lon in &[0.0, 90.0, -90.0, -180.0, 179.999] {
            let encoded = encode_longitude(lon);
            let decoded = decode_longitude(encoded);
            assert_in_delta!(decoded, lon, 8.4e-8);
        }
    }

    #[test]
    fn test_encode_latitude_ordering() {
        let south = encode_latitude(-45.0);
        let equator = encode_latitude(0.0);
        let north = encode_latitude(45.0);
        assert_lt!(south, equator);
        assert_lt!(equator, north);
    }

    #[test]
    fn test_encode_longitude_ordering() {
        let west = encode_longitude(-90.0);
        let prime = encode_longitude(0.0);
        let east = encode_longitude(90.0);
        assert_lt!(west, prime);
        assert_lt!(prime, east);
    }

    #[test]
    fn test_encode_latitude_boundary_90() {
        let encoded = encode_latitude(90.0);
        let decoded = decode_latitude(encoded);
        assert_lt!(decoded, 90.0);
        assert_gt!(decoded, 89.999);
    }

    #[test]
    fn test_encode_longitude_boundary_180() {
        let encoded = encode_longitude(180.0);
        let decoded = decode_longitude(encoded);
        assert_lt!(decoded, 180.0);
        assert_gt!(decoded, 179.999);
    }

    #[test]
    fn test_encode_latitude_min() {
        let encoded = encode_latitude(-90.0);
        let decoded = decode_latitude(encoded);
        assert_in_delta!(decoded, -90.0, 4.2e-8);
    }

    #[test]
    fn test_encode_longitude_min() {
        let encoded = encode_longitude(-180.0);
        let decoded = decode_longitude(encoded);
        assert_in_delta!(decoded, -180.0, 8.4e-8);
    }

    #[test]
    #[should_panic(expected = "latitude")]
    fn test_encode_latitude_out_of_range() {
        encode_latitude(91.0);
    }

    #[test]
    #[should_panic(expected = "longitude")]
    fn test_encode_longitude_out_of_range() {
        encode_longitude(181.0);
    }

    // Cross-validate with known Java outputs:
    // Java: GeoEncodingUtils.encodeLatitude(40.7128) = 971445247
    // Java: GeoEncodingUtils.encodeLongitude(-74.006) = -882925972
    #[test]
    fn test_encode_latitude_cross_validate() {
        let encoded = encode_latitude(40.7128);
        assert_eq!(encoded, 971_445_247);
    }

    #[test]
    fn test_encode_longitude_cross_validate() {
        let encoded = encode_longitude(-74.006);
        assert_eq!(encoded, -882_925_972);
    }
}
