# iOS CI/CD 设置指南

本文档指导如何配置 GitHub Actions 自动化发布到 TestFlight 和 App Store。

---

## 前置要求

1. **Apple Developer 账号**（已付费）
2. **App Store Connect API 密钥**
3. **iOS 分发证书和 Provisioning Profile**
4. **GitHub Repository Secrets**

---

## Step 1: 创建 App Store Connect API 密钥

1. 登录 [App Store Connect](https://appstoreconnect.apple.com)
2. 进入 **用户和访问** → **密钥**
3. 点击 **创建 API 密钥**
4. 选择 **"App Manager"** 角色
5. 下载生成的 `.p8` 密钥文件（只下载一次！）
6. 记录 **Key ID**（如 `8399F524UX`）和 **Issuer ID**（如 `3d8a8ce3-ad11-4ead-9e6c-38eecfe55269`）

---

## Step 2: 准备密钥文件

你需要获取以下文件的 Base64 编码：

### ASC API 密钥
```bash
base64 -i ~/.appstoreconnect/private_keys/AuthKey_8399F524UX.p8 | pbcopy
```

### 分发证书（.p12）
1. 在 Xcode 中导出 "Apple Distribution" 证书为 .p12 文件
2. 设置密码
3. 转换为 Base64：
```bash
base64 -i /path/to/Certificate.p12 | pbcopy
```

### Provisioning Profile
```bash
base64 -i ~/Library/MobileDevice/Provisioning\ Profiles/你的profile.mobileprovision | pbcopy
```

---

## Step 3: 配置 GitHub Secrets

进入 GitHub 仓库 → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**

添加以下 Secrets：

| Secret Name | 描述 | 获取方式 |
|------------|------|----------|
| `ASC_API_KEY_ID` | API Key ID | App Store Connect 密钥页面 |
| `ASC_API_KEY_ISSUER` | Issuer ID | App Store Connect 密钥页面 |
| `ASC_API_KEY_CONTENT` | .p8 文件内容 (Base64) | `base64 -i AuthKey_XXX.p8` |
| `IOS_DISTRIBUTION_CERTIFICATE_BASE64` | 证书 .p12 (Base64) | `base64 -i Certificate.p12` |
| `IOS_DISTRIBUTION_CERTIFICATE_PASSWORD` | 证书密码 | 导出时设置的密码 |
| `IOS_PROVISIONING_PROFILE_BASE64` | Profile (Base64) | `base64 -i profile.mobileprovision` |
| `KEYCHAIN_PASSWORD` | 临时 Keychain 密码 | 随机生成一个 |

---

## Step 4: 验证 Provisioning Profile

确保 `exportOptions.plist` 中的 Profile 名称正确：

```xml
<key>provisioningProfiles</key>
<dict>
    <key>com.fluxchi.app</key>
    <string>你的Profile名称</string>
</dict>
```

Profile 名称格式通常是：`XCWildcard` 或 `match AppStore com.fluxchi.app`

---

## Step 5: 更新 exportOptions.plist

编辑 `.github/workflows/exportOptions.plist`，将：

```xml
<string>XC_WILDCARD_替换为你的Profile名称</string>
```

替换为你的实际 Profile 名称。

---

## Workflow 触发条件

| 触发方式 | 操作 | 结果 |
|---------|------|------|
| Push to `main` | 正常提交代码 | 构建 + 上传 TestFlight |
| Pull Request | 创建 PR | 仅构建验证（不上传） |
| Tag `v*` | 创建 `v1.0.0` tag | 构建 + 上传 App Store |

---

## 手动触发

在 GitHub 仓库页面：

1. 进入 **Actions** 标签
2. 选择 **iOS Build & Deploy** workflow
3. 点击 **Run workflow**
4. 选择分支并运行

---

## 常见问题

### 1. 证书过期
- 证书有效期 1 年，到期前需要在 Xcode 中重新导出并更新 GitHub Secrets

### 2. Profile 过期
- 在 Apple Developer 后台重新下载 Profile
- 更新 `IOS_PROVISIONING_PROFILE_BASE64` Secret

### 3. API Key 失效
- 检查 Key ID 和 Issuer ID 是否正确
- 确认 .p8 文件没有损坏

### 4. 构建失败
- 检查 Xcode 版本是否支持
- 确认 Bundle ID 和 Team ID 匹配
- 查看 Actions 日志中的具体错误

---

## 快速命令参考

```bash
# 查看本地证书
security find-identity -v -p codesigning

# 查看 Profile 信息
security cms -D -i ~/Library/MobileDevice/Provisioning\ Profiles/xxx.mobileprovision

# 验证 Archive
xcodebuild -exportArchive -exportOptionsPth exportOptions.plist -archivePath FluxChi.xcarchive -exportPath ./export

# 手动上传 TestFlight
xcodebuild -uploadArchive -archivePath FluxChi.xcarchive -apiApiKey 8399F524UX -apiApiKeyIssuer 3d8a8ce3-ad11-4ead-9e6c-38eecfe55269
```

---

## 当前配置

- **Bundle ID**: `com.fluxchi.app`
- **Team ID**: `M4T239BM58`
- **Scheme**: `FluxChi`
- **Project**: `FluxChi.xcodeproj`
