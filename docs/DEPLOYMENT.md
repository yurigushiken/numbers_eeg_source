# GitHub Pages Deployment Guide

This guide walks you through deploying the EEG Source Localization website to GitHub Pages.

## 📋 Prerequisites

- [ ] All files in `docs/` directory are committed to your repository
- [ ] You have admin access to the GitHub repository
- [ ] Repository is public (or you have GitHub Pro for private repo Pages)

## 🚀 Step-by-Step Deployment

### Step 1: Verify File Structure

Make sure your repository has this structure:

```
numbers_eeg_source/
├── docs/
│   ├── index.html
│   ├── setup.html
│   ├── quickstart.html
│   ├── _config.yml
│   ├── css/
│   │   ├── style.css
│   │   └── setup.css
│   ├── js/
│   │   ├── main.js
│   │   └── setup.js
│   └── images/
│       ├── cursor.png
│       ├── terminal.png
│       └── terminal-with-env-activated.png
├── (other project files)
└── README.md
```

### Step 2: Add Screenshots

Before deploying, add the required images to `docs/images/`:

1. **cursor.png** - Screenshot of Cursor IDE interface
2. **terminal.png** - Screenshot of Cursor's integrated terminal
3. **terminal-with-env-activated.png** - Terminal showing `(numbers_eeg_source)` prompt

**How to take screenshots:**

**Windows:**
- Press `Windows + Shift + S` for Snipping Tool
- Select area and save

**macOS:**
- Press `Cmd + Shift + 4` to select area
- Or `Cmd + Shift + 5` for screenshot tools

**Linux:**
- Use your distribution's screenshot tool
- Or install `gnome-screenshot` / `flameshot`

### Step 3: Commit and Push

```bash
# Make sure you're in the repository root
cd numbers_eeg_source

# Add all docs files
git add docs/

# Commit
git commit -m "Add website documentation and GitHub Pages setup"

# Push to GitHub
git push origin main
```

### Step 4: Enable GitHub Pages

1. **Go to your repository on GitHub**
   - Navigate to: `https://github.com/YOUR_USERNAME/numbers_eeg_source`

2. **Open Settings**
   - Click on "Settings" tab at the top of the repository

3. **Navigate to Pages**
   - In the left sidebar, click "Pages" (under "Code and automation" section)

4. **Configure Source**
   - Under "Build and deployment":
     - **Source**: Select "Deploy from a branch"
     - **Branch**: Select `main` (or your default branch)
     - **Folder**: Select `/docs`
   - Click **Save**

5. **Wait for Deployment**
   - GitHub will start building your site
   - This usually takes 1-3 minutes
   - You'll see a message: "Your site is live at https://YOUR_USERNAME.github.io/numbers_eeg_source/"

### Step 5: Verify Deployment

1. **Check the Actions tab**
   - Go to the "Actions" tab in your repository
   - You should see a "pages build and deployment" workflow
   - Wait for it to complete (green checkmark)

2. **Visit your site**
   - Click the URL shown in Settings → Pages
   - Or go directly to: `https://YOUR_USERNAME.github.io/numbers_eeg_source/`

3. **Test all pages**
   - [ ] Homepage loads correctly
   - [ ] Setup page is accessible
   - [ ] Quickstart page works
   - [ ] All images display
   - [ ] Navigation works
   - [ ] CSS styles are applied
   - [ ] JavaScript interactions work

## 🔧 Troubleshooting

### Site not loading / 404 error

**Problem**: Page shows "404 There isn't a GitHub Pages site here"

**Solutions**:
1. Wait 5 minutes - initial deployment can take time
2. Check Settings → Pages shows "Your site is live"
3. Verify branch name is correct (main vs master)
4. Ensure `/docs` folder is selected, not root
5. Check that `index.html` exists in `docs/` directory

### CSS/JavaScript not loading

**Problem**: Site loads but has no styling or interactivity

**Solutions**:
1. Check browser console for errors (F12)
2. Verify file paths in HTML are relative (not absolute)
3. Ensure all CSS/JS files are in `docs/` subdirectories
4. Check file names match exactly (case-sensitive on Linux)

### Images not displaying

**Problem**: Broken image icons or missing screenshots

**Solutions**:
1. Verify images are in `docs/images/` directory
2. Check file names match exactly (case-sensitive)
3. Ensure images are committed and pushed to GitHub
4. Try hard refresh: Ctrl+Shift+R (Windows/Linux) or Cmd+Shift+R (macOS)

### Changes not appearing

**Problem**: Updated files but site still shows old content

**Solutions**:
1. Hard refresh your browser: Ctrl+Shift+R or Cmd+Shift+R
2. Clear browser cache
3. Check if changes are committed and pushed to GitHub
4. Wait for GitHub Actions to complete deployment
5. Check Actions tab for any deployment failures

### Wrong branch deploying

**Problem**: Old version of site is showing

**Solutions**:
1. Go to Settings → Pages
2. Verify correct branch is selected
3. Make sure latest changes are pushed to that branch
4. Check Actions tab to see which commit was deployed

## 🎨 Customization After Deployment

### Update site content

1. Edit files in `docs/` directory locally
2. Test changes locally (see docs/README.md)
3. Commit and push changes
4. GitHub automatically redeploys (takes 1-3 minutes)

### Custom domain (optional)

1. Buy a domain (e.g., from Namecheap, Google Domains)
2. In Settings → Pages, add your custom domain
3. Configure DNS records at your domain registrar:
   ```
   Type: CNAME
   Name: www
   Value: YOUR_USERNAME.github.io
   ```
4. Wait for DNS propagation (up to 24 hours)
5. Enable "Enforce HTTPS" in GitHub Pages settings

### Google Analytics (optional)

Add to `<head>` section of each HTML file:

```html
<!-- Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=GA_MEASUREMENT_ID"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'GA_MEASUREMENT_ID');
</script>
```

Replace `GA_MEASUREMENT_ID` with your Google Analytics ID.

## 📊 Monitoring

### Check site status

- **GitHub Status**: https://www.githubstatus.com/
- **Your deployment history**: Repository → Environments → github-pages

### View visitor analytics

GitHub Pages doesn't include built-in analytics. Options:
1. Google Analytics (free)
2. Plausible Analytics (privacy-focused)
3. Cloudflare Analytics (if using custom domain)

## 🔄 Continuous Deployment

Every time you push to the `main` branch, GitHub Pages automatically:
1. Detects changes in `docs/` directory
2. Rebuilds the site
3. Deploys new version
4. Updates live site (1-3 minutes)

**No manual deployment needed!**

## 📝 Checklist: First Deployment

- [ ] All HTML/CSS/JS files are in `docs/` directory
- [ ] All required images are added to `docs/images/`
- [ ] Files are committed to Git
- [ ] Changes are pushed to GitHub
- [ ] GitHub Pages enabled in Settings
- [ ] Branch set to `main` and folder set to `/docs`
- [ ] Waited 3 minutes for initial deployment
- [ ] Visited site URL and verified it works
- [ ] Tested all pages (index, setup, quickstart)
- [ ] Verified all images load
- [ ] Tested navigation between pages
- [ ] Checked console for JavaScript errors
- [ ] Tested on mobile device (responsive design)

## 🆘 Getting Help

If you encounter issues:

1. **Check GitHub Pages documentation**: https://docs.github.com/en/pages
2. **Check deployment logs**: Repository → Actions tab
3. **Browser console**: F12 → Console tab (for JavaScript errors)
4. **Network tab**: F12 → Network tab (for failed file loads)

## 🎉 Success!

Once deployed, share your site:
- Direct link: `https://YOUR_USERNAME.github.io/numbers_eeg_source/`
- Update main README.md with the link
- Add to your lab's website
- Share with research assistants

**Your EEG analysis pipeline now has professional documentation accessible to everyone!**
