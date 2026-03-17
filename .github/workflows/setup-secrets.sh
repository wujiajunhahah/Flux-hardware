#!/bin/bash
# 快速生成 GitHub Secrets 需要的 Base64 编码

echo "==================================="
echo "iOS CI/CD Secrets 生成工具"
echo "==================================="
echo ""

# 1. ASC API Key Content
echo "📝 1. ASC API Key Content (.p8 文件):"
echo "base64 -i ~/.appstoreconnect/private_keys/AuthKey_8399F524UX.p8 | pbcopy"
base64 -i ~/.appstoreconnect/private_keys/AuthKey_8399F524UX.p8
echo ""
echo ""

# 2. 询问证书路径
echo "📝 2. iOS 分发证书 (.p12 文件):"
echo "请提供证书路径 (或按 Enter 跳过):"
read CERT_PATH
if [ -n "$CERT_PATH" ]; then
    echo "base64 -i \"$CERT_PATH\" | pbcopy"
    base64 -i "$CERT_PATH"
    echo ""
fi

# 3. 询问 Profile 路径
echo "📝 3. Provisioning Profile:"
echo "请提供 Profile 路径 (或按 Enter 跳过):"
read PROFILE_PATH
if [ -n "$PROFILE_PATH" ]; then
    echo "base64 -i \"$PROFILE_PATH\" | pbcopy"
    base64 -i "$PROFILE_PATH"
    echo ""
fi

echo "==================================="
echo "📋 需要手动添加的 GitHub Secrets:"
echo "==================================="
echo ""
echo "ASC_API_KEY_ID=8399F524UX"
echo "ASC_API_KEY_ISSUER=3d8a8ce3-ad11-4ead-9e6c-38eecfe55269"
echo "ASC_API_KEY_CONTENT=<上面的 base64 输出>"
echo "IOS_DISTRIBUTION_CERTIFICATE_BASE64=<证书 base64>"
echo "IOS_DISTRIBUTION_CERTIFICATE_PASSWORD=<你设置的密码>"
echo "IOS_PROVISIONING_PROFILE_BASE64=<Profile base64>"
echo "KEYCHAIN_PASSWORD=<随机生成一个密码>"
echo ""
echo "==================================="
echo "访问以下链接添加 Secrets:"
echo "https://github.com/wujiajunhahah/Flux-hardware/settings/secrets/actions"
echo "==================================="
