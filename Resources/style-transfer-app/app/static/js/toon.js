// Toon Style Transformer JavaScript
let currentFilename = null;
let stream = null;

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // File upload handler
    const fileInput = document.getElementById('fileInput');
    if (fileInput) {
        fileInput.addEventListener('change', function(e) {
            if (e.target.files.length > 0) {
                handleFile(e.target.files[0]);
            }
        });
    }
    
    // Add click handlers for upload options
    const uploadOptions = document.querySelectorAll('.upload-option');
    uploadOptions.forEach(option => {
        option.addEventListener('click', function(e) {
            if (this.querySelector('.upload-btn').textContent.includes('Browse')) {
                fileInput.click();
            } else {
                openCamera();
            }
        });
    });
    
    console.log('Toon page initialized');
});

// Camera Functions (same as monet.js)
function openCamera() {
    const cameraPreview = document.getElementById('cameraPreview');
    const uploadSection = document.querySelector('.upload-section');
    
    if (cameraPreview) cameraPreview.style.display = 'block';
    if (uploadSection) uploadSection.style.display = 'none';
    
    navigator.mediaDevices.getUserMedia({ 
        video: { 
            facingMode: 'environment',
            width: { ideal: 1280 },
            height: { ideal: 720 }
        } 
    })
    .then(function(mediaStream) {
        stream = mediaStream;
        const video = document.getElementById('video');
        if (video) {
            video.srcObject = stream;
            video.style.display = 'block';
        }
    })
    .catch(function(err) {
        alert("Camera error: " + err.message);
        closeCamera();
    });
}

function closeCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
    
    const cameraPreview = document.getElementById('cameraPreview');
    const uploadSection = document.querySelector('.upload-section');
    const video = document.getElementById('video');
    
    if (cameraPreview) cameraPreview.style.display = 'none';
    if (uploadSection) uploadSection.style.display = 'grid';
    if (video) video.style.display = 'none';
}

function capturePhoto() {
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    
    if (!video || !canvas) return;
    
    const context = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    canvas.toBlob(function(blob) {
        const file = new File([blob], "camera_photo.jpg", { type: "image/jpeg" });
        handleFile(file);
        closeCamera();
    }, 'image/jpeg', 0.95);
}

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please select an image file (JPEG, PNG, etc.).');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    // Show preview immediately
    const preview = document.getElementById('imagePreview');
    if (preview) {
        preview.src = URL.createObjectURL(file);
        preview.style.display = 'block';
    }
    
    // Show original in results preview
    const originalImage = document.getElementById('originalImage');
    if (originalImage) {
        originalImage.src = URL.createObjectURL(file);
    }
    
    // Upload to server
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert('Upload error: ' + data.error);
            return;
        }
        
        currentFilename = data.filename;
        const convertBtn = document.getElementById('convertBtn');
        if (convertBtn) {
            convertBtn.disabled = false;
            convertBtn.innerHTML = '<i class="fas fa-play"></i> Convert to Toon Style';
        }
        
        alert('✅ Image uploaded successfully! Click "Convert to Toon Style" to transform it.');
    })
    .catch(error => {
        console.error('Upload error:', error);
        alert('Upload failed. Please try again.');
    });
}

function convertToToon() {
    if (!currentFilename) {
        alert('Please upload an image first.');
        return;
    }
    
    const loading = document.getElementById('loading');
    const convertBtn = document.getElementById('convertBtn');
    
    if (loading) loading.style.display = 'block';
    if (convertBtn) {
        convertBtn.disabled = true;
        convertBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
    }
    
    fetch('/convert/toon', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ filename: currentFilename })
    })
    .then(response => response.json())
    .then(data => {
        if (loading) loading.style.display = 'none';
        if (convertBtn) {
            convertBtn.disabled = false;
            convertBtn.innerHTML = '<i class="fas fa-play"></i> Convert to Toon Style';
        }
        
        if (data.error) {
            alert('❌ Conversion failed: ' + data.error);
            return;
        }
        
        const toonResult = document.getElementById('toonResult');
        const results = document.getElementById('results');
        
        if (toonResult) {
            toonResult.src = `/image/toon/${data.output_filename}?t=${new Date().getTime()}`;
            toonResult.onload = () => {
                alert('✨ Toon conversion complete! Your cartoon character is ready.');
            };
        }
        
        if (results) {
            results.style.display = 'block';
            // Scroll to results
            results.scrollIntoView({ behavior: 'smooth' });
        }
    })
    .catch(error => {
        if (loading) loading.style.display = 'none';
        if (convertBtn) {
            convertBtn.disabled = false;
            convertBtn.innerHTML = '<i class="fas fa-play"></i> Convert to Toon Style';
        }
        console.error('Conversion error:', error);
        alert('❌ Conversion failed. Please try again.');
    });
}

function downloadResult(style) {
    const resultImg = document.getElementById(`${style}Result`);
    if (!resultImg || !resultImg.src) {
        alert('No result image available to download.');
        return;
    }
    
    const filename = resultImg.src.split('/').pop().split('?')[0];
    window.open(`/download/${style}/${filename}`, '_blank');
}