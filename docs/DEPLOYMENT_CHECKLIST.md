# 🚀 GitHub Pages Deployment Checklist

Use this checklist to deploy your website step by step.

## 📸 Pre-Deployment: Add Screenshots

- [ ] **cursor.png** - Take screenshot of Cursor IDE
  - Open Cursor IDE
  - Capture clean interface view
  - Save to `docs/images/cursor.png`

- [ ] **terminal.png** - Take screenshot of Cursor terminal
  - Open Cursor's integrated terminal (Ctrl/Cmd + `)
  - Capture terminal view
  - Save to `docs/images/terminal.png`

- [ ] **terminal-with-env-activated.png** - Environment activated
  - Activate environment: `conda activate numbers_eeg_source`
  - Capture terminal showing `(numbers_eeg_source)` prompt
  - Save to `docs/images/terminal-with-env-activated.png`

## 📝 Step 1: Verify Files

- [ ] Check all HTML files exist:
  - `docs/index.html`
  - `docs/setup.html`
  - `docs/quickstart.html`

- [ ] Check all CSS files exist:
  - `docs/css/style.css`
  - `docs/css/setup.css`

- [ ] Check all JS files exist:
  - `docs/js/main.js`
  - `docs/js/setup.js`

- [ ] Check all images exist:
  - `docs/images/cursor.png`
  - `docs/images/terminal.png`
  - `docs/images/terminal-with-env-activated.png`

- [ ] Check configuration exists:
  - `docs/_config.yml`

## 🧪 Step 2: Test Locally

- [ ] Open terminal in `docs/` directory
- [ ] Run: `python -m http.server 8000`
- [ ] Open browser to: `http://localhost:8000`
- [ ] Test homepage loads
- [ ] Test setup page loads
- [ ] Test quickstart page loads
- [ ] Test all navigation links work
- [ ] Test platform tabs work (Windows/macOS/Linux)
- [ ] Test code copy functionality
- [ ] Test checklist persistence (check boxes, refresh, verify saved)
- [ ] Test on mobile view (F12 → Toggle device toolbar)
- [ ] Check browser console for errors (F12 → Console tab)
- [ ] Stop local server (Ctrl+C)

## 💾 Step 3: Commit and Push

Open terminal in repository root:

```bash
# Check status
git status

# Add all docs files
git add docs/

# Add website summary
git add WEBSITE_COMPLETION_SUMMARY.md

# Commit
git commit -m "Add website documentation with dark rainbow theme

- Complete homepage with features and overview
- Full setup guide with 10 detailed steps
- Quickstart guide for first analysis
- Interactive platform tabs and checklists
- Dark mode rainbow theme matching design requirements
- Ready for GitHub Pages deployment"

# Push to GitHub
git push origin main
```

- [ ] Files added with `git add`
- [ ] Changes committed with descriptive message
- [ ] Changes pushed to GitHub
- [ ] Verified push was successful (check GitHub.com)

## ⚙️ Step 4: Enable GitHub Pages

1. **Navigate to Repository Settings**
   - [ ] Go to: `https://github.com/YOUR_USERNAME/numbers_eeg_source`
   - [ ] Click "Settings" tab

2. **Configure Pages**
   - [ ] Click "Pages" in left sidebar (under "Code and automation")
   - [ ] Under "Source": Select "Deploy from a branch"
   - [ ] Under "Branch": Select `main` (or your default branch)
   - [ ] Under "Folder": Select `/docs`
   - [ ] Click "Save"

3. **Wait for Deployment**
   - [ ] Note the URL shown: `https://YOUR_USERNAME.github.io/numbers_eeg_source/`
   - [ ] Click "Actions" tab at top
   - [ ] Wait for "pages build and deployment" workflow to complete
   - [ ] Green checkmark indicates success (usually 1-3 minutes)

## ✅ Step 5: Verify Deployment

- [ ] Visit: `https://YOUR_USERNAME.github.io/numbers_eeg_source/`
- [ ] Homepage loads correctly
- [ ] All CSS styling applied (dark rainbow theme)
- [ ] All three images display
- [ ] Navigation bar works
- [ ] Click "Pre-Meeting Setup" → setup.html loads
- [ ] Click "Quickstart" → quickstart.html loads
- [ ] Platform tabs switch correctly
- [ ] Code blocks are copyable (click to test)
- [ ] Checklist items are clickable
- [ ] Test on mobile device or responsive mode
- [ ] Open browser console (F12) → No errors

## 📱 Step 6: Test on Different Devices

- [ ] Test on desktop browser (Chrome/Firefox/Edge/Safari)
- [ ] Test on mobile phone (actual device if possible)
- [ ] Test on tablet (if available)
- [ ] Test responsive breakpoints:
  - [ ] Desktop (>968px)
  - [ ] Tablet (768px-968px)
  - [ ] Mobile (<768px)

## 🔄 Step 7: Make Updates (If Needed)

If you need to fix something:

1. **Edit files locally**
   - [ ] Make changes to files in `docs/` directory
   - [ ] Test locally again

2. **Commit and push**
   ```bash
   git add docs/
   git commit -m "Fix: description of what you fixed"
   git push origin main
   ```

3. **Wait for auto-deployment**
   - [ ] GitHub Actions automatically rebuilds (1-3 minutes)
   - [ ] Hard refresh browser: Ctrl+Shift+R or Cmd+Shift+R
   - [ ] Verify changes appear

## 📢 Step 8: Share Your Website

- [ ] Copy website URL: `https://YOUR_USERNAME.github.io/numbers_eeg_source/`
- [ ] Update main `README.md` with website link
- [ ] Share with lab members
- [ ] Share with research assistants
- [ ] Add to lab's main website (if applicable)
- [ ] Bookmark for easy access

## 🎉 Completion

- [ ] All checklist items completed
- [ ] Website is live and accessible
- [ ] Tested and working correctly
- [ ] Shared with team

**Congratulations! Your EEG Source Localization Pipeline website is now live! 🚀**

---

## 🆘 Troubleshooting Quick Reference

### Site not loading (404)
→ Wait 5 minutes, check Settings → Pages shows "Your site is live"

### CSS not loading
→ Check browser console (F12), verify file paths are relative

### Images not displaying
→ Verify images exist in `docs/images/`, check file names (case-sensitive)

### Changes not appearing
→ Hard refresh: Ctrl+Shift+R (Windows/Linux) or Cmd+Shift+R (macOS)

### Need more help?
→ See `docs/DEPLOYMENT.md` for detailed troubleshooting

---

**Document Version**: 1.0
**Last Updated**: 2025-10-22
**Status**: Ready for use
