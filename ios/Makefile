# ─────────────────────────────────────────────────
# FluxChi iOS — Build & Distribution Pipeline
# ─────────────────────────────────────────────────
#
# Usage:
#   make build          — 编译 (Debug)
#   make archive        — 归档 (Release)
#   make export         — 导出 IPA (App Store)
#   make upload         — 上传到 App Store Connect
#   make testflight     — archive + export + upload 一键发布 TestFlight
#   make device         — 编译并安装到连接的 iPhone
#   make clean          — 清理构建产物
#   make bump           — 自增 build number
#   make status         — 显示当前签名/版本信息
#
# ─────────────────────────────────────────────────

# Project
SCHEME         := FluxChi
PROJECT        := FluxChi.xcodeproj
CONFIGURATION  := Release
DESTINATION    := generic/platform=iOS
DEVICE_DEST    := $(shell xcrun devicectl list devices 2>/dev/null | grep -m1 'iPhone' | awk '{print $$NF}')

# Paths
BUILD_DIR      := build
ARCHIVE_PATH   := $(BUILD_DIR)/FluxChi.xcarchive
EXPORT_DIR     := $(BUILD_DIR)/export
IPA_PATH       := $(EXPORT_DIR)/FluxChi.ipa
EXPORT_OPTIONS := Distribution/ExportOptions-appstore.plist

# App Store Connect API
ASC_KEY_ID     := 8399F524UX
ASC_ISSUER     := 3d8a8ce3-ad11-4ead-9e6c-38eecfe55269
ASC_KEY_PATH   := $(HOME)/.appstoreconnect/private_keys/AuthKey_$(ASC_KEY_ID).p8

# ─────────────────────────────────────────────────
# Targets
# ─────────────────────────────────────────────────

.PHONY: build archive export upload testflight device clean bump status help

help: ## 显示帮助
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

build: ## 编译 Debug
	@echo "🔨 Building $(SCHEME) (Debug)..."
	@xcodebuild build \
		-project $(PROJECT) \
		-scheme $(SCHEME) \
		-configuration Debug \
		-destination '$(DESTINATION)' \
		-quiet
	@echo "✅ Build succeeded"

archive: ## 归档 Release
	@echo "📦 Archiving $(SCHEME)..."
	@mkdir -p $(BUILD_DIR)
	@xcodebuild archive \
		-project $(PROJECT) \
		-scheme $(SCHEME) \
		-configuration $(CONFIGURATION) \
		-destination '$(DESTINATION)' \
		-archivePath $(ARCHIVE_PATH) \
		-allowProvisioningUpdates \
		-authenticationKeyPath $(ASC_KEY_PATH) \
		-authenticationKeyID $(ASC_KEY_ID) \
		-authenticationKeyIssuerID $(ASC_ISSUER) \
		-quiet
	@echo "✅ Archive: $(ARCHIVE_PATH)"

export: ## 导出 IPA
	@echo "📤 Exporting IPA..."
	@mkdir -p $(EXPORT_DIR)
	@xcodebuild -exportArchive \
		-archivePath $(ARCHIVE_PATH) \
		-exportPath $(EXPORT_DIR) \
		-exportOptionsPlist $(EXPORT_OPTIONS) \
		-allowProvisioningUpdates \
		-authenticationKeyPath $(ASC_KEY_PATH) \
		-authenticationKeyID $(ASC_KEY_ID) \
		-authenticationKeyIssuerID $(ASC_ISSUER) \
		-quiet
	@echo "✅ IPA: $(EXPORT_DIR)"

upload: ## 上传到 App Store Connect
	@echo "🚀 Uploading to App Store Connect..."
	@xcrun altool --upload-app \
		-f $(EXPORT_DIR)/*.ipa \
		--type ios \
		--apiKey $(ASC_KEY_ID) \
		--apiIssuer $(ASC_ISSUER) \
		2>&1 || \
	xcrun notarytool submit $(EXPORT_DIR)/*.ipa \
		--key $(ASC_KEY_PATH) \
		--key-id $(ASC_KEY_ID) \
		--issuer $(ASC_ISSUER) \
		--wait
	@echo "✅ Upload complete — check App Store Connect for processing"

testflight: archive export upload ## 一键发布 TestFlight
	@echo ""
	@echo "🎉 TestFlight build submitted!"
	@echo "   → App Store Connect 处理后自动可用"

device: ## 编译并安装到 iPhone
	@echo "📱 Building and installing to device..."
	@xcodebuild build \
		-project $(PROJECT) \
		-scheme $(SCHEME) \
		-configuration Debug \
		-destination 'platform=iOS,id=$(DEVICE_DEST)' \
		-quiet
	@echo "✅ Installed to device"

clean: ## 清理
	@echo "🧹 Cleaning..."
	@xcodebuild clean -project $(PROJECT) -scheme $(SCHEME) -quiet 2>/dev/null || true
	@rm -rf $(BUILD_DIR)
	@rm -rf ~/Library/Developer/Xcode/DerivedData/FluxChi-*
	@echo "✅ Clean"

bump: ## 自增 build number
	@CURRENT=$$(xcodebuild -project $(PROJECT) -scheme $(SCHEME) -showBuildSettings 2>/dev/null | \
		grep CURRENT_PROJECT_VERSION | tr -d ' ' | cut -d= -f2); \
	NEXT=$$((CURRENT + 1)); \
	sed -i '' "s/CURRENT_PROJECT_VERSION = $$CURRENT/CURRENT_PROJECT_VERSION = $$NEXT/g" $(PROJECT)/project.pbxproj; \
	echo "✅ Build number: $$CURRENT → $$NEXT"

status: ## 显示项目状态
	@echo "── FluxChi Project Status ──"
	@echo ""
	@echo "Scheme:     $(SCHEME)"
	@echo "Team ID:    M4T239BM58"
	@echo "Bundle ID:  com.fluxchi.app"
	@echo ""
	@echo "Signing identities:"
	@security find-identity -v -p codesigning 2>/dev/null | head -5
	@echo ""
	@echo "Version:"
	@xcodebuild -project $(PROJECT) -scheme $(SCHEME) -showBuildSettings 2>/dev/null | \
		grep -E 'MARKETING_VERSION|CURRENT_PROJECT_VERSION' | head -4
	@echo ""
	@echo "API Key:    $(ASC_KEY_ID)"
	@echo "Key file:   $(ASC_KEY_PATH)"
	@test -f $(ASC_KEY_PATH) && echo "Key status:  ✅ exists" || echo "Key status:  ❌ missing"
	@echo ""
	@echo "Connected devices:"
	@xcrun devicectl list devices 2>/dev/null | grep -E 'iPhone|iPad' | head -3 || echo "  (none)"
