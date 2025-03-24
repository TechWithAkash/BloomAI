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
//             toastr.error('Please select an image to upload.');
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
//             toastr.success('Image analyzed successfully!');
//         } catch (error) {
//             if (error.message.includes('Please log in')) {
//                 toastr.error('Please log in to access this feature.');
//                 window.location.href = '/login';
//             } else {
//                 errorMessage.textContent = error.message;
//                 errorMessage.style.display = 'block';
//                 toastr.error(error.message);
//             }
//         } finally {
//             analyzeButton.disabled = false;
//             loading.style.display = 'none';
//         }
//     });
// });



document.addEventListener('DOMContentLoaded', function() {
    // Navigation toggle
    const navToggle = document.getElementById('navToggle');
    const navLinks = document.querySelector('.nav-links');
    
    if (navToggle) {
        navToggle.addEventListener('click', function() {
            navLinks.classList.toggle('active');
            navToggle.classList.toggle('active');
        });
    }

    // FAQ toggles
    const faqItems = document.querySelectorAll('.faq-item');
    
    faqItems.forEach(item => {
        const question = item.querySelector('.faq-question');
        const answer = item.querySelector('.faq-answer');
        const toggleIcon = item.querySelector('.toggle-icon');
        
        question.addEventListener('click', () => {
            const expanded = item.classList.contains('active');
            
            // Close all FAQ items
            faqItems.forEach(i => {
                i.classList.remove('active');
                i.querySelector('.toggle-icon').textContent = '+';
            });
            
            // If clicked item wasn't expanded, open it
            if (!expanded) {
                item.classList.add('active');
                toggleIcon.textContent = '−';
            }
        });
    });

    // Image upload functionality
    const uploadForm = document.getElementById('upload-form');
    const imageInput = document.getElementById('image-input');
    const uploadArea = document.getElementById('uploadArea');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const loading = document.getElementById('loading');
    const resultSection = document.getElementById('result-section');
    const errorMessage = document.getElementById('error-message');

    if (uploadArea) {
        uploadArea.addEventListener('click', function() {
            imageInput.click();
        });
        
        uploadArea.addEventListener('dragover', function(e) {
            e.preventDefault();
            uploadArea.classList.add('drag-over');
        });
        
        uploadArea.addEventListener('dragleave', function() {
            uploadArea.classList.remove('drag-over');
        });
        
        uploadArea.addEventListener('drop', function(e) {
            e.preventDefault();
            uploadArea.classList.remove('drag-over');
            
            if (e.dataTransfer.files.length) {
                imageInput.files = e.dataTransfer.files;
                updatePreview();
            }
        });
        
        imageInput.addEventListener('change', function() {
            updatePreview();
        });
        
        function updatePreview() {
            const file = imageInput.files[0];
            
            if (file) {
                const reader = new FileReader();
                
                reader.onload = function(e) {
                    imagePreview.src = e.target.result;
                    uploadArea.style.display = 'none';
                    previewContainer.style.display = 'block';
                }
                
                reader.readAsDataURL(file);
            }
        }
    }

    // Form submission for image prediction
    if (uploadForm) {
        uploadForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const formData = new FormData(uploadForm);
            
            // Show loading
            loading.style.display = 'flex';
            resultSection.style.display = 'none';
            errorMessage.style.display = 'none';
            
            fetch(uploadForm.action, {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                loading.style.display = 'none';
                
                if (data.error) {
                    errorMessage.textContent = data.error;
                    errorMessage.style.display = 'block';
                    return;
                }
                
                // Display result
                document.getElementById('uploaded-image').src = '/' + data.image_path;
                document.getElementById('prediction-class').textContent = data.class;
                document.getElementById('prediction-confidence').textContent = data.confidence;
                
                const confidencePercent = parseFloat(data.confidence) * 100;
                document.getElementById('confidence-level').style.width = `${confidencePercent}%`;
                
                resultSection.style.display = 'block';
                
                // If heatmap is available
                if (data.heatmap) {
                    const heatmapSection = document.createElement('div');
                    heatmapSection.className = 'heatmap-section';
                    heatmapSection.innerHTML = `
                        <h4>AI Focus Areas</h4>
                        <img src="data:image/jpeg;base64,${data.heatmap}" alt="Heatmap">
                        <p>Colored areas show where the AI focused to make its prediction</p>
                    `;
                    document.querySelector('.result-details').appendChild(heatmapSection);
                }
                
                // If top predictions available
                if (data.top_predictions && Array.isArray(data.top_predictions)) {
                    const topPredictionsSection = document.createElement('div');
                    topPredictionsSection.className = 'top-predictions-section';
                    
                    let predictionsHTML = '<h4>Top Predictions</h4><ul>';
                    data.top_predictions.forEach(prediction => {
                        const flowerClass = prediction[0];
                        const confidence = (prediction[1] * 100).toFixed(2);
                        predictionsHTML += `
                            <li>
                                <span class="prediction-name">${flowerClass}</span>
                                <div class="mini-confidence-bar">
                                    <div class="mini-confidence-level" style="width: ${confidence}%"></div>
                                </div>
                                <span class="mini-percentage">${confidence}%</span>
                            </li>
                        `;
                    });
                    predictionsHTML += '</ul>';
                    
                    topPredictionsSection.innerHTML = predictionsHTML;
                    document.querySelector('.result-details').appendChild(topPredictionsSection);
                }
                
                // Add favorite button
                const favoriteButton = document.createElement('button');
                favoriteButton.className = 'favorite-button';
                favoriteButton.innerHTML = '<i class="far fa-heart"></i> Add to Favorites';
                favoriteButton.onclick = function() {
                    addToFavorites(data.class, favoriteButton);
                };
                document.querySelector('.result-details').appendChild(favoriteButton);
                
                // Add link to visualization
                if (data.identification_id) {
                    const visualizationLink = document.createElement('a');
                    visualizationLink.href = `/visualization/${data.identification_id}`;
                    visualizationLink.className = 'visualization-link';
                    visualizationLink.innerHTML = '<i class="fas fa-chart-pie"></i> Detailed Analysis';
                    document.querySelector('.result-details').appendChild(visualizationLink);
                }
            })
            .catch(error => {
                loading.style.display = 'none';
                errorMessage.textContent = 'An error occurred. Please try again.';
                errorMessage.style.display = 'block';
                console.error('Error:', error);
            });
        });
    }

    

    // Function to add a flower to favorites
    function addToFavorites(flowerClass, button) {
        fetch('/add_favorite', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ flower_class: flowerClass })
        })
        .then(response => response.json())
        .then(data => {
            if (data.message) {
                toastr.success(data.message);
                button.innerHTML = '<i class="fas fa-heart"></i> Added to Favorites';
                button.disabled = true;
            }
        })
        .catch(error => {
            console.error('Error:', error);
            toastr.error('Error adding to favorites');
        });
    }

    // Initialize favorite buttons on history page
    const favoriteButtons = document.querySelectorAll('.favorite-btn');
    if (favoriteButtons) {
        favoriteButtons.forEach(button => {
            button.addEventListener('click', function() {
                const flowerClass = this.getAttribute('data-flower');
                addToFavorites(flowerClass, this);
            });
        });
    }
});