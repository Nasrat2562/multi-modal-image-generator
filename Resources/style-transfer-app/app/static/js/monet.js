// app/static/js/monet.js
let currentFile = null;
let stream = null;

console.log("Monet page initialized");

// File upload handling
document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('fileInput');
    const browseBtn = document.querySelector('.upload-option:first-child .upload-btn');
    const cameraBtn = document.querySelector('.upload-option:nth-child(2) .upload-btn');
    
    browseBtn.addEventListener('click', () => fileInput.click());
    cameraBtn.addEventListener('click', openCamera);
    
    fileInput.addEventListener('change', handleFileSelect);
});

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    if (!file.type.startsWith('image/')) {
        alert('Please select an image file (JPG, PNG, etc.)');
        return;
    }
    
    // Upload to server
    uploadFile(file);
}

function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert('Upload failed: ' + data.error);
            return;
        }
        
        currentFile = data.filename;
        
        // Show preview
        const preview = document.getElementById('imagePreview');
        preview.src = URL.createObjectURL(file);
        preview.style.display = 'block';
        
        // Update original image in results
        document.getElementById('originalImage').src = preview.src;
        
        // Enable convert button
        document.getElementById('convertBtn').disabled = false;
        
        alert('Image uploaded successfully! Click "Convert to Monet Style" to begin.');
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Upload failed: ' + error);
    });
}

async function openCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ 
            video: { facingMode: 'user' } 
        });
        
        const video = document.getElementById('video');
        video.srcObject = stream;
        
        document.getElementById('cameraPreview').style.display = 'block';
        document.querySelector('.upload-section').style.display = 'none';
    } catch (err) {
        console.error('Camera error:', err);
        alert('Unable to access camera. Please make sure permissions are granted.');
    }
}

function closeCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
    
    document.getElementById('cameraPreview').style.display = 'none';
    document.querySelector('.upload-section').style.display = 'flex';
}

function capturePhoto() {
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const context = canvas.getContext('2d');
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    canvas.toBlob(function(blob) {
        const file = new File([blob], 'camera_photo.jpg', { type: 'image/jpeg' });
        uploadFile(file);
        closeCamera();
    }, 'image/jpeg');
}

function convertToMonet() {
    if (!currentFile) {
        alert('Please upload an image first.');
        return;
    }
    
    // Show loading
    document.getElementById('loading').style.display = 'block';
    document.getElementById('convertBtn').disabled = true;
    document.querySelector('.action-section').style.display = 'none';
    
    fetch('/convert/monet', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ filename: currentFile })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('loading').style.display = 'none';
        
        if (data.error) {
            alert('Conversion failed: ' + data.error);
            document.getElementById('convertBtn').disabled = false;
            document.querySelector('.action-section').style.display = 'block';
            return;
        }
        
        // Show result
        const resultImg = document.getElementById('monetResult');
        resultImg.src = `/image/monet/${data.output_filename}?t=${new Date().getTime()}`;
        
        document.getElementById('results').style.display = 'block';
        
        // Scroll to results
        document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
    })
    .catch(error => {
        document.getElementById('loading').style.display = 'none';
        console.error('Error:', error);
        alert('Conversion failed: ' + error);
        document.getElementById('convertBtn').disabled = false;
        document.querySelector('.action-section').style.display = 'block';
    });
}

function downloadResult(style) {
    const resultImg = document.getElementById(`${style}Result`);
    const filename = resultImg.src.split('/').pop().split('?')[0];
    window.open(`/download/${style}/${filename}`, '_blank');
}