//! Semantic dependency input, independent of Cargo's package-manifest rewriting.

use std::collections::BTreeMap;

/// The fields Cargo preserves when it rewrites a path/inline dependency into a
/// published table. Paths, package prose and Cargo's generated comments are not
/// dependency semantics. Git/alternate registries are refused: this contract is
/// for the suite's crates.io dependencies, not an assertion about arbitrary sources.
pub fn dependency_contract(manifest: &str) -> Result<String, String> {
    let parsed: toml::Value = manifest.parse().map_err(|e| format!("manifest: {e}"))?;
    let dependencies = parsed
        .get("dependencies")
        .and_then(toml::Value::as_table)
        .ok_or("missing dependencies")?;
    let mut canonical = BTreeMap::new();
    for name in [
        "sharpebench-core",
        "sharpebench-protocol",
        "sharpebench-sim",
    ] {
        let dependency = dependencies
            .get(name)
            .ok_or_else(|| format!("missing {name}"))?;
        let version = dependency
            .as_str()
            .or_else(|| dependency.get("version").and_then(toml::Value::as_str))
            .ok_or_else(|| format!("{name} needs an exact version"))?;
        let release = version
            .strip_prefix('=')
            .ok_or_else(|| format!("{name} must be exact-pinned"))?;
        // Current suite pins are numeric releases, not ranges or wildcard patches.
        if release.split('.').count() != 3
            || release
                .split('.')
                .any(|part| part.is_empty() || !part.bytes().all(|b| b.is_ascii_digit()))
        {
            return Err(format!("{name} needs an exact =major.minor.patch release"));
        }
        if let Some(table) = dependency.as_table() {
            for key in table.keys() {
                if ![
                    "version",
                    "path",
                    "features",
                    "default-features",
                    "optional",
                ]
                .contains(&key.as_str())
                {
                    return Err(format!("unsupported dependency field {name}.{key}"));
                }
            }
        }
        let flag = |key: &str, default: bool| -> Result<bool, String> {
            match dependency.get(key) {
                None => Ok(default),
                Some(value) => value
                    .as_bool()
                    .ok_or_else(|| format!("{name}.{key} must be boolean")),
            }
        };
        let mut features = match dependency.get("features") {
            None => Vec::new(),
            Some(value) => value
                .as_array()
                .ok_or_else(|| format!("{name}.features must be an array"))?
                .iter()
                .map(|v| {
                    v.as_str()
                        .map(String::from)
                        .ok_or_else(|| format!("{name} feature must be a string"))
                })
                .collect::<Result<Vec<_>, _>>()?,
        };
        features.sort();
        features.dedup();
        let mut record = toml::map::Map::new();
        record.insert("version".into(), toml::Value::String(version.into()));
        record.insert(
            "default-features".into(),
            toml::Value::Boolean(flag("default-features", true)?),
        );
        record.insert(
            "optional".into(),
            toml::Value::Boolean(flag("optional", false)?),
        );
        record.insert(
            "features".into(),
            toml::Value::Array(features.into_iter().map(toml::Value::String).collect()),
        );
        canonical.insert(name.to_string(), toml::Value::Table(record));
    }
    Ok(toml::Value::Table(canonical.into_iter().collect()).to_string())
}
