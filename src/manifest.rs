use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::{Path, PathBuf},
};

pub const MODEL_VERSION: &str = "ppocrv5-mobile-v1";
pub const RUNTIME_VERSION: &str = "1.22.1";

const RELEASE_BASE: &str = "https://github.com/reevebyte/LumenOCR/releases/download/assets-v1";
const MODELSCOPE_BASE: &str = "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0";

pub const DET_MODEL_FILE: &str = "ch_PP-OCRv5_det_mobile.onnx";
pub const CLS_MODEL_FILE: &str = "ch_PP-LCNet_x0_25_textline_ori_cls_mobile.onnx";
pub const REC_MODEL_FILE: &str = "ch_PP-OCRv5_rec_mobile.onnx";
pub const DICT_FILE: &str = "ppocrv5_dict.txt";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum AssetKind {
    DirectFile,
    /// Extract a single file whose archive path ends with `member_suffix`.
    TarGzMember {
        member_suffix: String,
    },
    /// Extract a single file whose archive path ends with `member_suffix`.
    ZipMember {
        member_suffix: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct AssetSpec {
    /// Target filename stored inside the local OCR directory.
    pub file_name: String,
    /// Candidate download URLs tried in order.
    pub urls: Vec<String>,
    /// Expected SHA-256 hex digest of the installed target file.
    pub sha256: String,
    /// Approximate download/installed size in bytes for progress reporting.
    pub expected_bytes: u64,
    pub kind: AssetKind,
    /// Whether this asset is the native ONNX Runtime shared library.
    #[serde(default)]
    pub is_runtime: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct Manifest {
    pub model_version: String,
    pub assets: Vec<AssetSpec>,
}

pub fn current_platform_supported() -> bool {
    platform_runtime_asset(std::env::consts::OS, std::env::consts::ARCH).is_some()
}

pub fn runtime_library_name() -> &'static str {
    runtime_library_name_for(std::env::consts::OS)
}

pub fn runtime_library_name_for(os: &str) -> &'static str {
    match os {
        "windows" => "onnxruntime.dll",
        "macos" => "libonnxruntime.dylib",
        _ => "libonnxruntime.so",
    }
}

pub fn override_runtime_dylib() -> Option<PathBuf> {
    std::env::var_os("LUMEN_ORT_DYLIB")
        .or_else(|| std::env::var_os("SCREENTRANS_ORT_DYLIB"))
        .map(PathBuf::from)
        .filter(|path| path.is_file())
}

fn platform_runtime_asset(os: &str, arch: &str) -> Option<AssetSpec> {
    let (file_name, member_suffix, sha256, expected_bytes): (
        &'static str,
        &'static str,
        &'static str,
        u64,
    ) = match (os, arch) {
        ("windows", "x86_64") => (
            "onnxruntime.dll",
            "runtimes/win-x64/native/onnxruntime.dll",
            "7788f3f38e9a339003f7d7e1bf47f928287cf409bb5273273149e1282fbf503f",
            12_416_032,
        ),
        ("linux", "x86_64") => (
            "libonnxruntime.so",
            "runtimes/linux-x64/native/libonnxruntime.so",
            "3907398e408dae083deb3439e8f643d9e26180ed614b29cc7d5ec342ce5ce06f",
            21_087_472,
        ),
        ("macos", "aarch64") => (
            "libonnxruntime.dylib",
            "runtimes/osx-arm64/native/libonnxruntime.dylib",
            "36a46aeb65eee6618daded125577192514d058d0cce6adc4892bc1bcc77cb9be",
            33_548_864,
        ),
        ("macos", "x86_64") => (
            "libonnxruntime.dylib",
            "runtimes/osx-x64/native/libonnxruntime.dylib",
            "9f2f73ddae8cfb7b5d9be7b84f0bfc3db07e5883d5baa78c4b658ec022b46275",
            37_740_976,
        ),
        _ => return None,
    };

    let release_direct = format!("{RELEASE_BASE}/{os}-{arch}-{file_name}");
    let nuget_archive = format!(
        "https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime/{RUNTIME_VERSION}/microsoft.ml.onnxruntime.{RUNTIME_VERSION}.nupkg"
    );

    Some(AssetSpec {
        file_name: file_name.into(),
        urls: vec![release_direct, nuget_archive],
        sha256: sha256.into(),
        expected_bytes,
        kind: AssetKind::ZipMember {
            member_suffix: member_suffix.into(),
        },
        is_runtime: true,
    })
}

pub fn default_manifest() -> Option<Manifest> {
    if let Ok(path) = std::env::var("LUMEN_OCR_MANIFEST")
        .or_else(|_| std::env::var("SCREENTRANS_LOCAL_OCR_MANIFEST"))
        && let Ok(bytes) = fs::read(path)
        && let Ok(parsed) = serde_json::from_slice::<Manifest>(&bytes)
    {
        return Some(parsed);
    }
    manifest_for(std::env::consts::OS, std::env::consts::ARCH)
}

pub fn manifest_for(os: &str, arch: &str) -> Option<Manifest> {
    let runtime = platform_runtime_asset(os, arch)?;
    let mut assets = Vec::with_capacity(5);
    if override_runtime_dylib().is_none() {
        assets.push(runtime);
    }
    assets.extend([
        AssetSpec {
            file_name: DET_MODEL_FILE.into(),
            urls: vec![
                format!("{MODELSCOPE_BASE}/onnx/PP-OCRv5/det/{DET_MODEL_FILE}"),
                format!("{RELEASE_BASE}/{DET_MODEL_FILE}"),
            ],
            sha256: "4d97c44a20d30a81aad087d6a396b08f786c4635742afc391f6621f5c6ae78ae".into(),
            expected_bytes: 4_819_576,
            kind: AssetKind::DirectFile,
            is_runtime: false,
        },
        AssetSpec {
            file_name: CLS_MODEL_FILE.into(),
            urls: vec![
                format!("{MODELSCOPE_BASE}/onnx/PP-OCRv5/cls/{CLS_MODEL_FILE}"),
                format!("{RELEASE_BASE}/{CLS_MODEL_FILE}"),
            ],
            sha256: "54379ae5174d026780215fc748a7f31910dee36818e63d49e17dc598ecc82df7".into(),
            expected_bytes: 1_018_508,
            kind: AssetKind::DirectFile,
            is_runtime: false,
        },
        AssetSpec {
            file_name: REC_MODEL_FILE.into(),
            urls: vec![
                format!("{MODELSCOPE_BASE}/onnx/PP-OCRv5/rec/{REC_MODEL_FILE}"),
                format!("{RELEASE_BASE}/{REC_MODEL_FILE}"),
            ],
            sha256: "5825fc7ebf84ae7a412be049820b4d86d77620f204a041697b0494669b1742c5".into(),
            expected_bytes: 16_631_306,
            kind: AssetKind::DirectFile,
            is_runtime: false,
        },
        AssetSpec {
            file_name: DICT_FILE.into(),
            urls: vec![
                format!("{MODELSCOPE_BASE}/paddle/PP-OCRv5/rec/ch_PP-OCRv5_rec_mobile/{DICT_FILE}"),
                format!("{RELEASE_BASE}/{DICT_FILE}"),
            ],
            sha256: "d1979e9f794c464c0d2e0b70a7fe14dd978e9dc644c0e71f14158cdf8342af1b".into(),
            expected_bytes: 74_012,
            kind: AssetKind::DirectFile,
            is_runtime: false,
        },
    ]);
    Some(Manifest {
        model_version: MODEL_VERSION.into(),
        assets,
    })
}

pub fn resolve_runtime_dylib(dir: &Path) -> Option<PathBuf> {
    if let Some(overridden) = override_runtime_dylib() {
        return Some(overridden);
    }
    let path = dir.join(runtime_library_name());
    path.is_file().then_some(path)
}

pub fn is_installed(dir: &Path, manifest: &Manifest) -> bool {
    if resolve_runtime_dylib(dir).is_none() {
        return false;
    }
    manifest.assets.iter().all(|asset| {
        if asset.is_runtime && override_runtime_dylib().is_some() {
            return true;
        }
        dir.join(&asset.file_name).is_file()
    })
}

pub fn installed_bytes(dir: &Path, manifest: &Manifest) -> u64 {
    manifest
        .assets
        .iter()
        .filter_map(|asset| fs::metadata(dir.join(&asset.file_name)).ok())
        .map(|meta| meta.len())
        .sum()
}

pub fn total_expected_bytes(manifest: &Manifest) -> u64 {
    manifest
        .assets
        .iter()
        .map(|asset| asset.expected_bytes)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_covers_windows_linux_and_macos_targets() {
        for (os, arch, expected_lib) in [
            ("windows", "x86_64", "onnxruntime.dll"),
            ("linux", "x86_64", "libonnxruntime.so"),
            ("macos", "aarch64", "libonnxruntime.dylib"),
            ("macos", "x86_64", "libonnxruntime.dylib"),
        ] {
            let manifest = manifest_for(os, arch).expect("supported target");
            assert_eq!(manifest.model_version, MODEL_VERSION);
            assert!(
                manifest
                    .assets
                    .iter()
                    .any(|a| a.file_name == DET_MODEL_FILE)
            );
            assert!(
                manifest
                    .assets
                    .iter()
                    .any(|a| a.file_name == REC_MODEL_FILE)
            );
            assert!(
                manifest
                    .assets
                    .iter()
                    .any(|a| a.file_name == CLS_MODEL_FILE)
            );
            assert!(manifest.assets.iter().any(|a| a.file_name == DICT_FILE));
            assert_eq!(runtime_library_name_for(os), expected_lib);
            for asset in &manifest.assets {
                assert_eq!(asset.sha256.len(), 64, "{os}/{arch}/{}", asset.file_name);
                assert!(
                    asset.sha256.chars().all(|c| c.is_ascii_hexdigit()),
                    "{os}/{arch}/{}",
                    asset.file_name
                );
                assert!(!asset.urls.is_empty());
            }
        }
        assert!(manifest_for("freebsd", "x86_64").is_none());
    }
}
