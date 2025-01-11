// static/js/main.js

// Navigation Toggle
document.addEventListener('DOMContentLoaded', function() {
    const navToggle = document.getElementById('navToggle');
    const navLinks = document.querySelector('.nav-links');

    navToggle.addEventListener('click', function() {
        this.classList.toggle('active');
        navLinks.classList.toggle('active');
    });

    // Close menu when clicking a link
    document.querySelectorAll('.nav-links a').forEach(link => {
        link.addEventListener('click', () => {
            navToggle.classList.remove('active');
            navLinks.classList.remove('active');
        });
    });

    // FAQ Toggle
    document.querySelectorAll('.faq-question').forEach(question => {
        question.addEventListener('click', () => {
            const faqItem = question.parentElement;
            document.querySelectorAll('.faq-item').forEach(item => {
                if (item !== faqItem) {
                    item.classList.remove('active');
                }
            });
            faqItem.classList.toggle('active');
            question.querySelector('.toggle-icon').textContent = 
                faqItem.classList.contains('active') ? '-' : '+';
        });
    });

    // Drag and Drop Upload
    const uploadArea = document.getElementById('uploadArea');
    const imageInput = document.getElementById('image-input');

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, () => {
            uploadArea.classList.add('highlight');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, () => {
            uploadArea.classList.remove('highlight');
        });
    });

    uploadArea.addEventListener('drop', handleDrop);
    uploadArea.addEventListener('click', () => imageInput.click());

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        imageInput.files = files;
    }

    // Form Submission
    const uploadForm = document.getElementById('upload-form');
    const resultSection = document.getElementById('result-section');
    const errorMessage = document.getElementById('error-message');

    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const formData = new FormData();
        const fileInput = document.getElementById('image-input');

        if (!fileInput.files[0]) {
            showError('Please select an image file');
            return;
        }

        formData.append('file', fileInput.files[0]);
        showLoading();

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (response.ok) {
                showResult(result);
            } else {
                showError(result.error || 'An error occurred during analysis');
            }
        } catch (error) {
            showError('An error occurred while connecting to the server');
        } finally {
            hideLoading();
        }
    });

    function showLoading() {
        const loadingSpinner = document.createElement('div');
        loadingSpinner.className = 'loading-spinner';
        uploadForm.appendChild(loadingSpinner);
        uploadForm.querySelector('button').disabled = true;
    }

    function hideLoading() {
        const loadingSpinner = document.querySelector('.loading-spinner');
        if (loadingSpinner) {
            loadingSpinner.remove();
        }
        uploadForm.querySelector('button').disabled = false;
    }

    function showResult(result) {
        document.getElementById('uploaded-image').src = '/' + result.image_path;
        document.getElementById('prediction-class').textContent = result.class;
        document.getElementById('prediction-confidence').textContent = result.confidence;
        
        const confidenceLevel = document.getElementById('confidence-level');
        const confidenceValue = parseFloat(result.confidence) || 0;
        confidenceLevel.style.width = `${confidenceValue}%`;

        resultSection.style.display = 'block';
        errorMessage.style.display = 'none';
        
        resultSection.scrollIntoView({ behavior: 'smooth' });
    }

    function showError(message) {
        errorMessage.textContent = message;
        errorMessage.style.display = 'block';
        resultSection.style.display = 'none';
    }
});