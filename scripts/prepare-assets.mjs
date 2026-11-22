#!/usr/bin/env node
/**
 * Prepare and verify assets for the `assets-v1` GitHub Release of LumenOCR.
 *
 * Downloads PP-OCRv5 Mobile ONNX models from ModelScope (RapidAI/RapidOCR)
 * and the official Microsoft.ML.OnnxRuntime NuGet package, extracts each
 * platform's single required shared library, writes `build/assets-v1/checksums.json`,
 * and with `--update-manifest` updates pinned SHA-256 digests in `src/manifest.rs`.
 */
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { mkdir, readFile, stat, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = fileURLToPath(new URL('../', import.meta.url));
const OUT_DIR = resolve(ROOT, 'build/assets-v1');
const MANIFEST_RS = resolve(ROOT, 'src/manifest.rs');
const ORT_VERSION = '1.22.1';
const NUGET_URL = `https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime/${ORT_VERSION}/microsoft.ml.onnxruntime.${ORT_VERSION}.nupkg`;
const MODELSCOPE_BASE = 'https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0';

const MODELS = [
  {
    file: 'ch_PP-OCRv5_det_mobile.onnx',
    url: `${MODELSCOPE_BASE}/onnx/PP-OCRv5/det/ch_PP-OCRv5_det_mobile.onnx`,
    sha256: '4d97c44a20d30a81aad087d6a396b08f786c4635742afc391f6621f5c6ae78ae',
  },
  {
    file: 'ch_PP-LCNet_x0_25_textline_ori_cls_mobile.onnx',
    url: `${MODELSCOPE_BASE}/onnx/PP-OCRv5/cls/ch_PP-LCNet_x0_25_textline_ori_cls_mobile.onnx`,
    sha256: '54379ae5174d026780215fc748a7f31910dee36818e63d49e17dc598ecc82df7',
  },
  {
    file: 'ch_PP-OCRv5_rec_mobile.onnx',
    url: `${MODELSCOPE_BASE}/onnx/PP-OCRv5/rec/ch_PP-OCRv5_rec_mobile.onnx`,
    sha256: '5825fc7ebf84ae7a412be049820b4d86d77620f204a041697b0494669b1742c5',
  },
  {
    file: 'ppocrv5_dict.txt',
    url: `${MODELSCOPE_BASE}/paddle/PP-OCRv5/rec/ch_PP-OCRv5_rec_mobile/ppocrv5_dict.txt`,
    sha256: 'd1979e9f794c464c0d2e0b70a7fe14dd978e9dc644c0e71f14158cdf8342af1b',
  },
];

const RUNTIMES = [
  {
    target: 'windows-x86_64',
    os: 'windows',
    arch: 'x86_64',
    libName: 'onnxruntime.dll',
    outFile: 'windows-x86_64-onnxruntime.dll',
    memberPatterns: [/^runtimes\/win-x64\/native\/onnxruntime\.dll$/],
  },
  {
    target: 'linux-x86_64',
    os: 'linux',
    arch: 'x86_64',
    libName: 'libonnxruntime.so',
    outFile: 'linux-x86_64-libonnxruntime.so',
    memberPatterns: [/^runtimes\/linux-x64\/native\/libonnxruntime\.so$/],
  },
  {
    target: 'macos-aarch64',
    os: 'macos',
    arch: 'aarch64',
    libName: 'libonnxruntime.dylib',
    outFile: 'macos-aarch64-libonnxruntime.dylib',
    memberPatterns: [
      /^runtimes\/osx-arm64\/native\/libonnxruntime\.dylib$/,
      /^runtimes\/osx\.11\.0-arm64\/native\/libonnxruntime\.dylib$/,
    ],
  },
  {
    target: 'macos-x86_64',
    os: 'macos',
    arch: 'x86_64',
    libName: 'libonnxruntime.dylib',
    outFile: 'macos-x86_64-libonnxruntime.dylib',
    memberPatterns: [
      /^runtimes\/osx-x64\/native\/libonnxruntime\.dylib$/,
      /^runtimes\/osx\.10\.14-x64\/native\/libonnxruntime\.dylib$/,
    ],
  },
];

async function sha256File(path) {
  const bytes = await readFile(path);
  return createHash('sha256').update(bytes).digest('hex');
}

function downloadWithCurl(url, dest) {
  execFileSync('curl', ['-fSL', '--retry', '3', '--connect-timeout', '20', '-o', dest, url], {
    stdio: 'inherit',
  });
}

function listZipEntries(zipPath) {
  return execFileSync('unzip', ['-Z1', zipPath], {
    encoding: 'utf8',
    maxBuffer: 16 * 1024 * 1024,
  })
    .split(/\r?\n/)
    .map(line => line.trim())
    .filter(Boolean);
}

async function updateManifestRs(summary) {
  let content = await readFile(MANIFEST_RS, 'utf8');
  content = content.replace(
    /pub const RUNTIME_VERSION: &str = "[^"]+";/,
    `pub const RUNTIME_VERSION: &str = "${ORT_VERSION}";`,
  );

  const arms = RUNTIMES.map(rt => {
    const info = summary.runtimes[rt.target];
    return `        ("${rt.os}", "${rt.arch}") => (
            "${rt.libName}",
            "${info.member}",
            "${info.sha256}",
            ${info.bytes},
        ),`;
  }).join('\n');

  const replacement = `fn platform_runtime_asset(os: &str, arch: &str) -> Option<AssetSpec> {
    let (file_name, member_suffix, sha256, expected_bytes): (
        &'static str,
        &'static str,
        &'static str,
        u64,
    ) = match (os, arch) {
${arms}
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
}`;

  const pattern = /fn platform_runtime_asset\(os: &str, arch: &str\) -> Option<AssetSpec> \{[\s\S]*?\n\}/;
  if (!pattern.test(content)) {
    throw new Error('Could not find platform_runtime_asset() in src/manifest.rs');
  }
  content = content.replace(pattern, replacement);
  await writeFile(MANIFEST_RS, content);
  console.log(`Updated ${MANIFEST_RS}`);
}

async function main() {
  const updateManifest = process.argv.includes('--update-manifest');
  await mkdir(OUT_DIR, { recursive: true });
  const summary = { ortVersion: ORT_VERSION, nugetUrl: NUGET_URL, models: {}, runtimes: {} };

  for (const model of MODELS) {
    const dest = join(OUT_DIR, model.file);
    let ready = false;
    try {
      ready = (await sha256File(dest)) === model.sha256;
    } catch {}
    if (!ready) {
      console.log(`Downloading ${model.file}...`);
      downloadWithCurl(model.url, dest);
      const actual = await sha256File(dest);
      if (actual !== model.sha256) {
        throw new Error(`SHA-256 mismatch for ${model.file}: expected ${model.sha256}, got ${actual}`);
      }
    }
    const { size } = await stat(dest);
    summary.models[model.file] = { sha256: model.sha256, bytes: size };
  }

  const nupkgPath = join(OUT_DIR, `microsoft.ml.onnxruntime.${ORT_VERSION}.nupkg`);
  let haveNupkg = false;
  try {
    haveNupkg = (await stat(nupkgPath)).size > 1_000_000;
  } catch {}
  if (!haveNupkg) {
    console.log(`Downloading ${NUGET_URL}...`);
    downloadWithCurl(NUGET_URL, nupkgPath);
  }

  const entries = listZipEntries(nupkgPath);
  for (const rt of RUNTIMES) {
    const member = entries.find(entry => rt.memberPatterns.some(pattern => pattern.test(entry)));
    if (!member) {
      throw new Error(`Missing native library for ${rt.target} in ${nupkgPath}`);
    }
    const outPath = join(OUT_DIR, rt.outFile);
    const extracted = execFileSync('unzip', ['-p', nupkgPath, member], {
      maxBuffer: 128 * 1024 * 1024,
    });
    await writeFile(outPath, extracted);
    const sha256 = await sha256File(outPath);
    const { size } = await stat(outPath);
    summary.runtimes[rt.target] = { file: rt.outFile, member, sha256, bytes: size };
    console.log(`Prepared ${rt.outFile}: ${size} bytes, member=${member}, sha256=${sha256}`);
  }

  await writeFile(join(OUT_DIR, 'checksums.json'), JSON.stringify(summary, null, 2) + '\n');
  if (updateManifest) {
    await updateManifestRs(summary);
  }
  console.log(`\nDone. Upload the 8 assets from ${OUT_DIR} to GitHub Release 'assets-v1'.`);
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
