# Setting Up GitHub Repository

Your local git repository is ready! Follow these steps to create and push to GitHub:

## Option 1: Using GitHub Website (Recommended)

1. **Create a new repository on GitHub:**
   - Go to https://github.com/new
   - Repository name: `multilingual-intent-slot-filling` (or your preferred name)
   - Description: "Multilingual Intent Detection & Slot Filling using XLM-RoBERTa and CRF"
   - Choose **Public** or **Private**
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
   - Click "Create repository"

2. **Push your code to GitHub:**
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   git branch -M main
   git push -u origin main
   ```

   Replace `YOUR_USERNAME` and `YOUR_REPO_NAME` with your actual GitHub username and repository name.

## Option 2: Using GitHub CLI

If you have GitHub CLI installed:

```bash
gh repo create multilingual-intent-slot-filling --public --source=. --remote=origin --push
```

## Option 3: Using SSH (if you have SSH keys set up)

```bash
git remote add origin git@github.com:YOUR_USERNAME/YOUR_REPO_NAME.git
git branch -M main
git push -u origin main
```

## After Pushing

Your repository will be live on GitHub! You can:
- View it at: `https://github.com/YOUR_USERNAME/YOUR_REPO_NAME`
- Share the link with others
- Continue making commits and pushing updates

## Future Updates

To push future changes:
```bash
git add .
git commit -m "Your commit message"
git push
```

