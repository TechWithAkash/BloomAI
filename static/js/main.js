// document.addEventListener('DOMContentLoaded', function() {
//     const navToggle = document.getElementById('navToggle');
//     const navLinks = document.querySelector('.nav-links');
//     const uploadArea = document.getElementById('uploadArea');
//     const imageInput = document.getElementById('image-input');
//     const previewContainer = document.getElementById('preview-container');
//     const imagePreview = document.getElementById('image-preview');
//     const analyzeButton = document.getElementById('analyze-button');
//     const uploadForm = document.getElementById('upload-form');
//     const resultSection = document.getElementById('result-section');
//     const loading = document.getElementById('loading');
//     const errorMessage = document.getElementById('error-message');

//     // Navigation Toggle
//     navToggle.addEventListener('click', function() {
//         this.classList.toggle('active');
//         navLinks.classList.toggle('active');
//     });

//     // Close menu when clicking a link
//     document.querySelectorAll('.nav-links a').forEach(link => {
//         link.addEventListener('click', () => {
//             navToggle.classList.remove('active');
//             navLinks.classList.remove('active');
//         });
//     });

//     // FAQ Toggle
//     document.querySelectorAll('.faq-question').forEach(question => {
//         question.addEventListener('click', () => {
//             const faqItem = question.parentElement;
//             document.querySelectorAll('.faq-item').forEach(item => {
//                 if (item !== faqItem) {
//                     item.classList.remove('active');
//                 }
//             });
//             faqItem.classList.toggle('active');
//             question.querySelector('.toggle-icon').textContent = 
//                 faqItem.classList.contains('active') ? '-' : '+';
//         });
//     });

//     // Drag and Drop Upload
//     ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
//         uploadArea.addEventListener(eventName, preventDefaults, false);
//     });

//     function preventDefaults(e) {
//         e.preventDefault();
//         e.stopPropagation();
//     }

//     ['dragenter', 'dragover'].forEach(eventName => {
//         uploadArea.addEventListener(eventName, () => {
//             uploadArea.classList.add('highlight');
//         });
//     });

//     ['dragleave', 'drop'].forEach(eventName => {
//         uploadArea.addEventListener(eventName, () => {
//             uploadArea.classList.remove('highlight');
//         });
//     });

//     uploadArea.addEventListener('drop', handleDrop);
//     uploadArea.addEventListener('click', () => imageInput.click());

//     function handleDrop(e) {
//         const dt = e.dataTransfer;
//         const files = dt.files;
//         imageInput.files = files;
//         handleImageUpload(files[0]);
//     }

//     imageInput.addEventListener('change', (e) => {
//         if (e.target.files[0]) {
//             handleImageUpload(e.target.files[0]);
//         }
//     });

//     function handleImageUpload(file) {
//         const reader = new FileReader();
//         reader.onload = function(e) {
//             imagePreview.src = e.target.result;
//             previewContainer.style.display = 'block';
//             analyzeButton.style.display = 'block';
//             resultSection.style.display = 'none';
//             errorMessage.style.display = 'none';
//         }
//         reader.readAsDataURL(file);
//     }

//     uploadForm.addEventListener('submit', async (e) => {
//         e.preventDefault();

//         if (!imageInput.files.length) {
//             alert('Please select an image to upload.');
//             return;
//         }

//         const formData = new FormData();
//         formData.append('file', imageInput.files[0]);

//         analyzeButton.disabled = true;
//         loading.style.display = 'block';
//         resultSection.style.display = 'none';
//         errorMessage.style.display = 'none';

//         try {
//             const response = await fetch('/predict', {
//                 method: 'POST',
//                 body: formData
//             });

//             if (!response.ok) {
//                 const errorText = await response.text();
//                 throw new Error(errorText || 'Failed to analyze image');
//             }

//             const data = await response.json();

//             document.getElementById('uploaded-image').src = imagePreview.src;
//             document.getElementById('prediction-class').textContent = data.class;
//             document.getElementById('confidence-level').style.width = data.confidence;
//             document.getElementById('prediction-confidence').textContent = data.confidence;
//             resultSection.style.display = 'block';
//         } catch (error) {
//             if (error.message.includes('Please log in')) {
//                 alert('Please log in to access this feature.');
//                 window.location.href = '/login';
//             } else {
//                 errorMessage.textContent = error.message;
//                 errorMessage.style.display = 'block';
//             }
//         } finally {
//             analyzeButton.disabled = false;
//             loading.style.display = 'none';
//         }
//     });
// });
document.addEventListener('DOMContentLoaded', function() {
    const navToggle = document.getElementById('navToggle');
    const navLinks = document.querySelector('.nav-links');
    const uploadArea = document.getElementById('uploadArea');
    const imageInput = document.getElementById('image-input');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const analyzeButton = document.getElementById('analyze-button');
    const uploadForm = document.getElementById('upload-form');
    const resultSection = document.getElementById('result-section');
    const loading = document.getElementById('loading');
    const errorMessage = document.getElementById('error-message');

    // Navigation Toggle
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
        handleImageUpload(files[0]);
    }

    imageInput.addEventListener('change', (e) => {
        if (e.target.files[0]) {
            handleImageUpload(e.target.files[0]);
        }
    });

    function handleImageUpload(file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            imagePreview.src = e.target.result;
            previewContainer.style.display = 'block';
            analyzeButton.style.display = 'block';
            resultSection.style.display = 'none';
            errorMessage.style.display = 'none';
        }
        reader.readAsDataURL(file);
    }

    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        if (!imageInput.files.length) {
            toastr.error('Please select an image to upload.');
            return;
        }

        const formData = new FormData();
        formData.append('file', imageInput.files[0]);

        analyzeButton.disabled = true;
        loading.style.display = 'block';
        resultSection.style.display = 'none';
        errorMessage.style.display = 'none';

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(errorText || 'Failed to analyze image');
            }

            const data = await response.json();

            document.getElementById('uploaded-image').src = imagePreview.src;
            document.getElementById('prediction-class').textContent = data.class;
            document.getElementById('confidence-level').style.width = data.confidence;
            document.getElementById('prediction-confidence').textContent = data.confidence;
            resultSection.style.display = 'block';
            toastr.success('Image analyzed successfully!');
        } catch (error) {
            if (error.message.includes('Please log in')) {
                toastr.error('Please log in to access this feature.');
                window.location.href = '/login';
            } else {
                errorMessage.textContent = error.message;
                errorMessage.style.display = 'block';
                toastr.error(error.message);
            }
        } finally {
            analyzeButton.disabled = false;
            loading.style.display = 'none';
        }
    });
});