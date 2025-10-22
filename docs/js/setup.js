// ===================================================================
// SETUP PAGE JAVASCRIPT
// Handles platform tabs, troubleshooting accordions, and checklists
// ===================================================================

document.addEventListener('DOMContentLoaded', function() {

    // ===================================================================
    // PLATFORM TAB SWITCHING
    // ===================================================================

    // Initialize all tab containers
    const tabContainers = document.querySelectorAll('.platform-tabs');

    tabContainers.forEach(container => {
        const buttons = container.querySelectorAll('.tab-button');
        const contents = container.querySelectorAll('.tab-content');

        buttons.forEach(button => {
            button.addEventListener('click', function() {
                const platform = this.dataset.platform;

                // Remove active class from all buttons and contents in this container
                buttons.forEach(btn => btn.classList.remove('active'));
                contents.forEach(content => content.classList.remove('active'));

                // Add active class to clicked button
                this.classList.add('active');

                // Show corresponding content
                const targetContent = container.querySelector(`.tab-content[data-platform="${platform}"]`);
                if (targetContent) {
                    targetContent.classList.add('active');
                }
            });
        });

        // Activate first tab by default
        if (buttons.length > 0) {
            buttons[0].click();
        }
    });

    // ===================================================================
    // TROUBLESHOOTING ACCORDION
    // ===================================================================

    const troubleshootingItems = document.querySelectorAll('.troubleshooting-item');

    troubleshootingItems.forEach(item => {
        const header = item.querySelector('.troubleshooting-header');

        header.addEventListener('click', function() {
            // Toggle active state
            const isActive = item.classList.contains('active');

            // Close all other items
            troubleshootingItems.forEach(otherItem => {
                if (otherItem !== item) {
                    otherItem.classList.remove('active');
                }
            });

            // Toggle current item
            if (isActive) {
                item.classList.remove('active');
            } else {
                item.classList.add('active');
            }
        });
    });

    // ===================================================================
    // CHECKLIST PERSISTENCE
    // ===================================================================

    // Save checklist state to localStorage
    const checkboxes = document.querySelectorAll('.checklist-item input[type="checkbox"]');
    const storageKey = 'setup-checklist-state';

    // Load saved state
    const savedState = JSON.parse(localStorage.getItem(storageKey) || '{}');

    checkboxes.forEach((checkbox, index) => {
        // Set unique ID if not present
        if (!checkbox.id) {
            checkbox.id = `checklist-${index}`;
        }

        // Restore saved state
        if (savedState[checkbox.id]) {
            checkbox.checked = true;
        }

        // Save state on change
        checkbox.addEventListener('change', function() {
            savedState[this.id] = this.checked;
            localStorage.setItem(storageKey, JSON.stringify(savedState));

            // Update progress indicator if exists
            updateProgress();
        });
    });

    // ===================================================================
    // PROGRESS TRACKING
    // ===================================================================

    function updateProgress() {
        const total = checkboxes.length;
        const completed = Array.from(checkboxes).filter(cb => cb.checked).length;
        const percentage = Math.round((completed / total) * 100);

        // Update progress indicator (if we add one to the page)
        const progressIndicator = document.getElementById('setup-progress');
        if (progressIndicator) {
            progressIndicator.textContent = `${completed}/${total} tasks completed (${percentage}%)`;
            progressIndicator.style.color = percentage === 100 ? 'var(--green)' : 'var(--cyan)';
        }

        // Show celebration if completed
        if (percentage === 100) {
            showCelebration();
        }
    }

    // Initial progress update
    updateProgress();

    // ===================================================================
    // CELEBRATION ANIMATION
    // ===================================================================

    function showCelebration() {
        // Only show once per session
        if (sessionStorage.getItem('celebration-shown')) {
            return;
        }

        sessionStorage.setItem('celebration-shown', 'true');

        const celebration = document.createElement('div');
        celebration.innerHTML = '🎉 Setup Complete! You\'re ready to analyze EEG data!';
        celebration.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: linear-gradient(135deg, var(--green), var(--cyan));
            color: white;
            padding: 2rem 3rem;
            border-radius: 16px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
            font-size: 1.5rem;
            font-weight: 700;
            z-index: 10000;
            animation: celebrationPop 0.5s ease;
        `;

        document.body.appendChild(celebration);

        setTimeout(() => {
            celebration.style.animation = 'celebrationFade 0.5s ease';
            setTimeout(() => {
                document.body.removeChild(celebration);
            }, 500);
        }, 3000);
    }

    // Add celebration animations
    const celebrationStyles = document.createElement('style');
    celebrationStyles.textContent = `
        @keyframes celebrationPop {
            0% { transform: translate(-50%, -50%) scale(0); opacity: 0; }
            50% { transform: translate(-50%, -50%) scale(1.1); }
            100% { transform: translate(-50%, -50%) scale(1); opacity: 1; }
        }

        @keyframes celebrationFade {
            from { opacity: 1; transform: translate(-50%, -50%) scale(1); }
            to { opacity: 0; transform: translate(-50%, -50%) scale(0.8); }
        }
    `;
    document.head.appendChild(celebrationStyles);

    // ===================================================================
    // SMOOTH SCROLLING FOR STEP LINKS
    // ===================================================================

    const stepLinks = document.querySelectorAll('a[href^="#step-"]');

    stepLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            const targetElement = document.querySelector(targetId);

            if (targetElement) {
                targetElement.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });

                // Highlight the step briefly
                targetElement.style.animation = 'highlightStep 1s ease';
                setTimeout(() => {
                    targetElement.style.animation = '';
                }, 1000);
            }
        });
    });

    // Add highlight animation
    const highlightStyles = document.createElement('style');
    highlightStyles.textContent = `
        @keyframes highlightStep {
            0%, 100% { box-shadow: 0 0 0 rgba(6, 182, 212, 0); }
            50% { box-shadow: 0 0 40px rgba(6, 182, 212, 0.5); }
        }
    `;
    document.head.appendChild(highlightStyles);

    // ===================================================================
    // COPY CODE BLOCKS
    // ===================================================================

    const codeBlocks = document.querySelectorAll('.code-block code, .inline-code');

    codeBlocks.forEach(block => {
        block.style.cursor = 'pointer';
        block.title = 'Click to copy';

        block.addEventListener('click', function() {
            const text = this.textContent.trim();

            navigator.clipboard.writeText(text).then(() => {
                // Visual feedback
                const originalBg = this.style.backgroundColor;
                this.style.backgroundColor = 'rgba(16, 185, 129, 0.3)';

                setTimeout(() => {
                    this.style.backgroundColor = originalBg;
                }, 500);

                // Show toast
                showToast('Copied to clipboard!');
            }).catch(err => {
                showToast('Failed to copy', true);
            });
        });
    });

    // ===================================================================
    // TOAST NOTIFICATIONS
    // ===================================================================

    function showToast(message, isError = false) {
        const toast = document.createElement('div');
        toast.textContent = message;
        toast.style.cssText = `
            position: fixed;
            bottom: 2rem;
            right: 2rem;
            background: ${isError ? 'linear-gradient(135deg, var(--red), var(--orange))' : 'linear-gradient(135deg, var(--green), var(--cyan))'};
            color: white;
            padding: 1rem 2rem;
            border-radius: 8px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
            z-index: 1000;
            font-weight: 600;
            animation: slideIn 0.3s ease;
        `;

        document.body.appendChild(toast);

        setTimeout(() => {
            toast.style.animation = 'slideOut 0.3s ease';
            setTimeout(() => {
                document.body.removeChild(toast);
            }, 300);
        }, 2000);
    }

    // ===================================================================
    // KEYBOARD SHORTCUTS
    // ===================================================================

    let currentStep = 1;
    const totalSteps = document.querySelectorAll('.step-container').length;

    document.addEventListener('keydown', function(e) {
        // Alt + Arrow keys to navigate steps
        if (e.altKey) {
            if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
                e.preventDefault();
                navigateToStep(currentStep + 1);
            } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
                e.preventDefault();
                navigateToStep(currentStep - 1);
            }
        }
    });

    function navigateToStep(stepNumber) {
        if (stepNumber < 1 || stepNumber > totalSteps) {
            return;
        }

        currentStep = stepNumber;
        const stepElement = document.getElementById(`step-${stepNumber}`);

        if (stepElement) {
            stepElement.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });

            // Highlight briefly
            stepElement.style.animation = 'highlightStep 1s ease';
            setTimeout(() => {
                stepElement.style.animation = '';
            }, 1000);
        }
    }

    // Update current step based on scroll position
    window.addEventListener('scroll', function() {
        const steps = document.querySelectorAll('.step-container');
        const scrollPosition = window.scrollY + 200;

        steps.forEach((step, index) => {
            if (step.offsetTop <= scrollPosition) {
                currentStep = index + 1;
            }
        });
    });

    // ===================================================================
    // PLATFORM DETECTION & AUTO-SELECT
    // ===================================================================

    function detectPlatform() {
        const userAgent = window.navigator.userAgent.toLowerCase();

        if (userAgent.indexOf('mac') !== -1) {
            return 'macos';
        } else if (userAgent.indexOf('linux') !== -1) {
            return 'linux';
        } else {
            return 'windows';
        }
    }

    // Auto-select platform tab on page load
    const detectedPlatform = detectPlatform();

    tabContainers.forEach(container => {
        const platformButton = container.querySelector(`.tab-button[data-platform="${detectedPlatform}"]`);
        if (platformButton) {
            platformButton.click();
        }
    });

    // Show platform detection message
    const platformNames = {
        'windows': 'Windows',
        'macos': 'macOS',
        'linux': 'Linux'
    };

    showToast(`Detected ${platformNames[detectedPlatform]} - tabs auto-selected`);
});
