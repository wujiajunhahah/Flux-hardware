# iOS CI/CD Workflow

自动构建并发布到 TestFlight 和 App Store。

---

## 🚀 快速开始

### 1. 添加 GitHub Secrets

访问：https://github.com/wujiajunhahah/Flux-hardware/settings/secrets/actions

#### 已知的 ASC 密钥（可直接复制）：

| Secret Name | Value |
|------------|-------|
| `ASC_API_KEY_ID` | `8399F524UX` |
| `ASC_API_KEY_ISSUER` | `3d8a8ce3-ad11-4ead-9e6c-38eecfe55269` |
| `ASC_API_KEY_CONTENT` | `LS0tLS1CRUdJTiBQUklWQVRFIEtFWS0tLS0tCk1JR1RBZ0VBTUJNR0J5cUdTTTQ5QWdFR0NDcUdTTTQ5QXdFSEJIa3dkd0lCQVFRZ25IRy9lVHc1bFRBTFpPNFUKUjZJUExITGxjVlBYSHpyd2g1SnJGMVl6QWxlZ0NnWUlLb1pJemowREFRZWhSQU5DQUFRSmVMUjd3V1M0Ujh1QwpuSDNvT2dGZ1drSFVXOCtRd2UyZG8wYmVENnJjUE14a2xuL0lIR1NIcTNuczZTT3NuTWU4V0ZtS0t1UytkandYCmFsOXlKYUJzCi0tLS0tRU5EIFBSSVZBVEUgS0VZLS0tLS0=` |

#### 需要你准备的 Secrets：

| Secret Name | 获取方式 |
|------------|----------|
| `IOS_DISTRIBUTION_CERTIFICATE_BASE64` | Xcode 导出证书 .p12 → `base64 -i cert.p12` |
| `IOS_DISTRIBUTION_CERTIFICATE_PASSWORD` | 导出时设置的密码 |
| `IOS_PROVISIONING_PROFILE_BASE64` | Apple Developer 下载 → `base64 -i profile.mobileprovision` |
| `KEYCHAIN_PASSWORD` | 随机生成，如：`workflow-keychain-2024` |

### 2. 运行 Workflow

```bash
# Push to main → 自动上传 TestFlight
git push origin main

# 创建 tag → 上传 App Store
git tag v1.0.0
git push origin v1.0.0
```

---

## 📋 Workflow 说明

| Job | 触发条件 | 操作 |
|-----|---------|------|
| `build-testflight` | Push to `main` | 构建 + 上传 TestFlight |
| `build-pr` | Pull Request | 仅构建验证 |
| `release-appstore` | Tag `v*` | 构建 + 上传 App Store |

---

## 🔍 本地测试

```bash
# 本地构建验证
cd ios
xcodebuild archive \
  -project FluxChi.xcodeproj \
  -scheme FluxChi \
  -archivePath ./FluxChi.xcarchive \
  -destination 'generic/platform=iOS' \
  CODE_SIGN_STYLE=Automatic \
  DEVELOPMENT_TEAM=M4T239BM58 \
  PRODUCT_BUNDLE_IDENTIFIER=com.fluxchi.app
```

---

## 📱 当前配置

- **Bundle ID**: `com.fluxchi.app`
- **Team ID**: `M4T239BM58`
- **App ID**: `6760378928`
- **App Name**: FocuX
