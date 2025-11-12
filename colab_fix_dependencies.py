# ============================================================================
# GOOGLE COLAB - QUICK FIX CELL
# Run this cell FIRST before any other imports
# ============================================================================

# Upgrade qdrant-client to compatible version
print("⚙️  Upgrading qdrant-client...")
!pip install -q --upgrade "qdrant-client>=1.7.0"

# Upgrade transformers for Gemma-3n support
print("⚙️  Upgrading transformers...")
!pip install -q --upgrade "transformers>=4.53.0"

# Install other required packages
print("⚙️  Installing dependencies...")
!pip install -q python-dotenv pydantic-settings tiktoken fastembed

print("\n✅ Dependencies upgraded successfully!")
print("\n⚠️  IMPORTANT: After running this cell, restart the runtime!")
print("   Runtime → Restart runtime")
print("   Then continue with the next cells.")
