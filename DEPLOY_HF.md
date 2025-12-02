# Deploy to Hugging Face Spaces

Quick guide to deploy your app to Hugging Face Spaces.

## Method 1: Using Hugging Face Website (Easiest)

1. **Go to your Space**: https://huggingface.co/spaces/kitsakisG/Pneumonia-Detection

2. **Click "Files and versions"** tab

3. **Click "Add file" → "Upload files"**

4. **Upload these files** from your local `Healthcare-Detection` folder:
   - `app.py` (main app)
   - `app/streamlit_app.py`
   - `app/interactive_training.py` (NEW!)
   - `requirements.txt`
   - `.spaces/README.md` → rename to `README.md` in root
   - Entire `src/` folder

5. **Commit changes** - HF will auto-rebuild and deploy!

## Method 2: Using Git (Advanced)

### Step 1: Get Hugging Face Token

1. Go to https://huggingface.co/settings/tokens
2. Click "New token"
3. Name it: `github-actions-sync`
4. Type: **Write**
5. Copy the token (starts with `hf_...`)

### Step 2: Add Token to GitHub

1. Go to your GitHub repo: https://github.com/kitsakisGk/Healthcare-Detection/settings/secrets/actions
2. Click "New repository secret"
3. Name: `HF_TOKEN`
4. Value: Paste your HF token
5. Save

### Step 3: Manual Push from Terminal

```bash
cd /d/Projects/Healthcare-Detection

# Add HF as remote
git remote add hf https://huggingface.co/spaces/kitsakisG/Pneumonia-Detection

# Push (will ask for username and token)
git push hf main --force
```

**When prompted:**
- Username: `kitsakisG`
- Password: Your HF token (starts with `hf_...`)

## Method 3: Clone HF Space and Copy Files

```bash
# Clone your HF space
git clone https://huggingface.co/spaces/kitsakisG/Pneumonia-Detection hf-space
cd hf-space

# Copy files from Healthcare-Detection
cp ../Healthcare-Detection/app.py .
cp ../Healthcare-Detection/requirements.txt .
cp -r ../Healthcare-Detection/src .
cp -r ../Healthcare-Detection/app .
cp ../Healthcare-Detection/.spaces/README.md README.md

# Commit and push
git add .
git commit -m "Update with new interactive training feature"
git push
```

## Verify Deployment

1. Go to https://huggingface.co/spaces/kitsakisG/Pneumonia-Detection
2. Check "Logs" tab to see build progress
3. Wait 2-3 minutes for rebuild
4. App should load automatically!

## Troubleshooting

### "Welcome to Streamlit" showing

**Problem**: Default app showing instead of yours

**Fix**: Make sure these files are in the **root** of your HF Space:
- `app.py` (not in a folder!)
- `README.md` with correct `app_file: app.py`
- `requirements.txt`
- `src/` folder

### Build failing

**Problem**: Missing dependencies or files

**Fix**: Check "Logs" tab for errors
- Make sure all imports in `app.py` work
- Verify `requirements.txt` has all packages
- Check file paths are correct

### Models not loading

**Problem**: Model files too large for HF

**Fix**: Either:
1. Upload small demo models (<1GB)
2. Use untrained models for demo
3. Or show message "Please train your own model"

## Current Status

Your HF Space is configured for:
- **App file**: `app.py`
- **SDK**: Streamlit 1.28.0
- **Python**: 3.10

The easiest way is **Method 1** - just upload files via the web interface!
