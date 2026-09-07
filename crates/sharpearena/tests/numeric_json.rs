//! Isolated builds must parse the same state values as feature-unified builds.

#[test]
fn numeric_json_state_roundtrips_preserve_every_bit() {
    for i in 0_u64..2048 {
        let bits = i.wrapping_mul(0x9E37_79B9_7F4A_7C15) & 0x7FEF_FFFF_FFFF_FFFF;
        for x in [f64::from_bits(bits), -f64::from_bits(bits)] {
            let text = serde_json::to_string(&x).unwrap();
            assert_eq!(text.parse::<f64>().unwrap().to_bits(), x.to_bits());
            let parsed: f64 = serde_json::from_str(&text).unwrap();
            assert_eq!(
                parsed.to_bits(),
                x.to_bits(),
                "numeric JSON drift for {text}"
            );
        }
    }
}
