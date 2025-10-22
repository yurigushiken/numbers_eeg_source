# EEG Source Localization Pipeline - Website Documentation

This directory contains the GitHub Pages website for the EEG Source Localization Pipeline.

## 🌐 Website Structure

```
docs/
├── index.html          # Homepage with overview and features
├── setup.html          # Complete pre-meeting setup guide
├── quickstart.html     # Quick start guide for running first analysis
├── _config.yml         # GitHub Pages configuration
├── css/
│   ├── style.css       # Main stylesheet (dark rainbow theme)
│   └── setup.css       # Setup page specific styles
├── js/
│   ├── main.js         # Main JavaScript (navigation, animations)
│   └── setup.js        # Setup page interactivity (tabs, checklists)
└── images/
    ├── cursor.png                         # Cursor IDE screenshot
    ├── terminal.png                       # Terminal screenshot
    └── terminal-with-env-activated.png    # Environment activation screenshot
```

## 🎨 Design Theme

The website uses a **dark mode rainbow theme** inspired by GitHub's syntax highlighting:

- **Dark Background**: #0d1117 (GitHub dark)
- **Rainbow Colors**: Pink, Purple, Blue, Cyan, Green, Yellow, Orange, Red
- **Interactive Elements**: Smooth animations, gradient effects
- **Responsive Design**: Works on mobile, tablet, and desktop

## 📸 Required Screenshots

To complete the website, add these screenshots to the `images/` directory:

1. **cursor.png** - Screenshot of Cursor IDE
2. **terminal.png** - Screenshot of Cursor's integrated terminal
3. **terminal-with-env-activated.png** - Terminal showing `(numbers_eeg_source)` environment

### Additional Screenshots for setup.html

The setup guide has 24 `[SCREENSHOT: ...]` placeholders. Replace these with actual screenshots showing:
- Software installation screens
- Conda environment creation
- Data folder structure
- Configuration steps
- Verification outputs

## 🚀 Deploying to GitHub Pages

### Method 1: Through GitHub Settings (Recommended)

1. Push all files in the `docs/` directory to your GitHub repository
2. Go to repository **Settings** → **Pages**
3. Under "Source", select:
   - **Source**: Deploy from a branch
   - **Branch**: `main` (or your default branch)
   - **Folder**: `/docs`
4. Click **Save**
5. GitHub will build and deploy your site
6. Your site will be available at: `https://yourusername.github.io/numbers_eeg_source/`

### Method 2: Using GitHub Actions (Advanced)

Create `.github/workflows/pages.yml`:

```yaml
name: Deploy GitHub Pages

on:
  push:
    branches: ["main"]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

jobs:
  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4
      - name: Setup Pages
        uses: actions/configure-pages@v4
      - name: Upload artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: 'docs'
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
```

## 🛠️ Local Development

To test the website locally:

### Option A: Python HTTP Server

```bash
cd docs
python -m http.server 8000
```

Then open: http://localhost:8000

### Option B: Live Server (VS Code / Cursor)

1. Install "Live Server" extension
2. Right-click on `index.html`
3. Select "Open with Live Server"

## ✨ Features

### Interactive Elements

- **Platform Detection**: Automatically selects Windows/macOS/Linux tabs based on user's OS
- **Code Copy**: Click any code block to copy to clipboard
- **Smooth Scrolling**: Navigation links smoothly scroll to sections
- **Progress Tracking**: Setup checklist saves progress to browser storage
- **Keyboard Shortcuts**: Alt + Arrow keys to navigate between setup steps
- **Toast Notifications**: Visual feedback for actions
- **Easter Egg**: Konami code activates rainbow mode!

### Accessibility

- Semantic HTML structure
- ARIA labels where appropriate
- Keyboard navigation support
- High contrast color ratios
- Responsive font sizing

## 📝 Customization

### Changing Colors

Edit `css/style.css` and modify the CSS variables in `:root`:

```css
:root {
    --pink: #ff4d9d;
    --purple: #a855f7;
    --blue: #3b82f6;
    --cyan: #06b6d4;
    --green: #10b981;
    /* ... etc */
}
```

### Adding New Pages

1. Create new HTML file in `docs/`
2. Copy header/nav/footer structure from existing pages
3. Link stylesheets: `style.css` and `setup.css` (if needed)
4. Link scripts: `main.js` and `setup.js` (if needed)
5. Update navigation links in all pages

### Updating Content

- **Homepage**: Edit `index.html`
- **Setup Guide**: Edit `setup.html`
- **Quickstart**: Edit `quickstart.html`
- **Styles**: Edit `css/style.css` or `css/setup.css`
- **JavaScript**: Edit `js/main.js` or `js/setup.js`

## 📧 Contact

For issues or questions about the website:
- Open an issue on GitHub
- Email: your-lab@university.edu

## 📄 License

This website is part of the EEG Source Localization Pipeline project and is licensed under the MIT License.
