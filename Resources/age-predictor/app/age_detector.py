import cv2
import numpy as np
import os
import requests
import tempfile

class AgeDetector:
    def __init__(self):
        print("🤖 Advanced Age Detector Initialized")
        
        # Load multiple face detectors for better coverage
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_alt_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
        self.face_alt2_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
        self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        
        # Try to load DNN-based face detector (more accurate)
        self.dnn_net = self.load_dnn_detector()
        
    def load_dnn_detector(self):
        """Load DNN-based face detector for better accuracy"""
        try:
            # Download DNN model files if not exists
            proto_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
            model_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel"
            
            proto_path = "deploy.prototxt"
            model_path = "res10_300x300_ssd_iter_140000.caffemodel"
            
            if not os.path.exists(proto_path):
                print("Downloading DNN model...")
                self.download_file(proto_url, proto_path)
            if not os.path.exists(model_path):
                self.download_file(model_url, model_path)
                
            net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
            print("✓ DNN Face Detector loaded")
            return net
        except Exception as e:
            print(f"⚠ DNN detector not available: {e}, using Haar cascades")
            return None
    
    def download_file(self, url, filename):
        """Download file from URL"""
        try:
            response = requests.get(url, stream=True)
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        except Exception as e:
            print(f"Download failed: {e}")
            raise
    
    def detect_faces_advanced(self, img):
        """Advanced face detection using multiple methods"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization for better contrast in low light
        gray_eq = cv2.equalizeHist(gray)
        
        all_faces = []
        
        # Method 1: DNN-based detection (most accurate)
        dnn_faces = self.detect_faces_dnn(img)
        if dnn_faces:
            all_faces.extend(dnn_faces)
        
        # Method 2: Multiple Haar cascades with different parameters
        haar_faces = self.detect_faces_haar(gray_eq)
        if haar_faces:
            all_faces.extend(haar_faces)
        
        # Method 3: Original image (no equalization)
        haar_original = self.detect_faces_haar(gray)
        if haar_original:
            all_faces.extend(haar_original)
        
        # Remove duplicates and return
        return self.remove_duplicate_faces(all_faces)
    
    def detect_faces_dnn(self, img):
        """DNN-based face detection - works better with angles and lighting"""
        if self.dnn_net is None:
            return []
            
        try:
            (h, w) = img.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, 
                                        (300, 300), (104.0, 177.0, 123.0))
            self.dnn_net.setInput(blob)
            detections = self.dnn_net.forward()
            
            faces = []
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                
                # Filter weak detections
                if confidence > 0.5:  # Lower threshold for better detection
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    
                    # Ensure coordinates are within image bounds
                    startX, startY = max(0, startX), max(0, startY)
                    endX, endY = min(w, endX), min(h, endY)
                    
                    width = endX - startX
                    height = endY - startY
                    
                    if width > 30 and height > 30:  # Reasonable minimum size
                        faces.append((startX, startY, width, height))
            
            return faces
        except Exception as e:
            print(f"DNN detection error: {e}")
            return []
    
    def detect_faces_haar(self, gray):
        """Enhanced Haar cascade detection with multiple classifiers and parameters"""
        faces = []
        
        # Different classifiers for various angles
        classifiers = [
            (self.face_cascade, 1.1, 3, (30, 30)),      # Front face
            (self.face_alt_cascade, 1.1, 3, (30, 30)),  # Alternative front
            (self.face_alt2_cascade, 1.1, 3, (30, 30)), # Another alternative
            (self.profile_cascade, 1.1, 3, (30, 30)),   # Profile faces
        ]
        
        # Try different scale factors and min neighbors
        scale_factors = [1.1, 1.2, 1.3]
        min_neighbors_list = [3, 4, 5]
        
        for cascade, base_scale, base_neighbors, min_size in classifiers:
            for scale in scale_factors:
                for min_neighbors in min_neighbors_list:
                    try:
                        detected = cascade.detectMultiScale(
                            gray,
                            scaleFactor=scale,
                            minNeighbors=min_neighbors,
                            minSize=min_size,
                            flags=cv2.CASCADE_SCALE_IMAGE
                        )
                        if len(detected) > 0:
                            faces.extend(detected)
                    except Exception:
                        continue
        
        return faces
    
    def preprocess_image(self, img):
        """Preprocess image for better face detection in various conditions"""
        # Convert to different color spaces
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        
        # Try different channels
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        v_channel = hsv[:,:,2]  # Value channel from HSV
        l_channel = lab[:,:,0]  # Lightness channel from LAB
        
        # Apply different enhancements
        gray_eq = cv2.equalizeHist(gray)
        v_eq = cv2.equalizeHist(v_channel)
        l_eq = cv2.equalizeHist(l_channel)
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray_clahe = clahe.apply(gray)
        
        return [gray, gray_eq, v_eq, l_eq, gray_clahe]
    
    def detect_faces_all_methods(self, img):
        """Try face detection on multiple preprocessed versions"""
        all_faces = []
        processed_images = self.preprocess_image(img)
        
        for processed_img in processed_images:
            # Try DNN on original color image
            dnn_faces = self.detect_faces_dnn(img)
            if dnn_faces:
                all_faces.extend(dnn_faces)
            
            # Try Haar on processed image
            haar_faces = self.detect_faces_haar(processed_img)
            if haar_faces:
                all_faces.extend(haar_faces)
        
        return self.remove_duplicate_faces(all_faces)
    
    def remove_duplicate_faces(self, faces):
        """Remove overlapping face detections"""
        if not faces:
            return []
        
        # Convert to list of tuples
        faces = [tuple(face) for face in faces]
        
        # Sort by area (largest first)
        faces.sort(key=lambda x: x[2]*x[3], reverse=True)
        
        filtered_faces = []
        for i, (x, y, w, h) in enumerate(faces):
            overlap = False
            for j, (x2, y2, w2, h2) in enumerate(filtered_faces):
                if i != j:
                    # Calculate overlap
                    dx = min(x+w, x2+w2) - max(x, x2)
                    dy = min(y+h, y2+h2) - max(y, y2)
                    if dx > 0 and dy > 0:
                        overlap_area = dx * dy
                        area1 = w * h
                        area2 = w2 * h2
                        # If overlap is more than 40% of smaller face, consider duplicate
                        if overlap_area > 0.4 * min(area1, area2):
                            overlap = True
                            break
            if not overlap:
                filtered_faces.append((x, y, w, h))
        
        return filtered_faces
    
    def predict_age(self, image_path, output_dir="results"):
        """Advanced age estimation with robust face detection"""
        try:
            # Create directories
            os.makedirs(output_dir, exist_ok=True)
            
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                return {'success': False, 'error': 'Could not read image file'}
            
            # Resize if too large
            height, width = img.shape[:2]
            if width > 1200:
                scale = 1200 / width
                new_width = 1200
                new_height = int(height * scale)
                img = cv2.resize(img, (new_width, new_height))
            
            print("🔍 Detecting faces with advanced methods...")
            
            # Use advanced face detection
            faces = self.detect_faces_all_methods(img)
            
            if len(faces) == 0:
                # Final attempt with more aggressive parameters
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray_eq = cv2.equalizeHist(gray)
                
                # Try very aggressive detection
                aggressive_faces = self.face_cascade.detectMultiScale(
                    gray_eq, scaleFactor=1.05, minNeighbors=2, minSize=(20, 20)
                )
                
                if len(aggressive_faces) > 0:
                    faces = self.remove_duplicate_faces(aggressive_faces)
            
            if len(faces) == 0:
                error_msg = """No face detected. Please ensure:
• Face is clearly visible
• Good lighting is helpful but not required
• Looking straight at camera works best
• Try different angles if front view fails
• Remove sunglasses or heavy obstructions"""
                return {'success': False, 'error': error_msg}
            
            print(f"✅ Found {len(faces)} face(s)")
            
            age_estimates = []
            confidence_scores = []
            
            for i, (x, y, w, h) in enumerate(faces):
                # Extract face region
                face_roi = img[y:y+h, x:x+w]
                
                # Advanced age estimation
                age, confidence = self.estimate_age_advanced(face_roi, w, h, img.shape)
                age_estimates.append(age)
                confidence_scores.append(confidence)
                
                # Draw detailed annotations
                self.draw_face_analysis(img, x, y, w, h, age, confidence, i+1)
            
            # Calculate weighted average
            if confidence_scores:
                total_confidence = sum(confidence_scores)
                weighted_age = sum(age * conf for age, conf in zip(age_estimates, confidence_scores)) / total_confidence
                final_age = int(round(weighted_age))
            else:
                final_age = int(sum(age_estimates) / len(age_estimates))
            
            # Add final result
            self.draw_final_result(img, final_age, len(faces), np.mean(confidence_scores))
            
            # Save result
            original_name = os.path.basename(image_path)
            name, ext = os.path.splitext(original_name)
            output_filename = f"age_result_{name}{ext}"
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, img)
            
            return {
                'success': True,
                'age': final_age,
                'faces_detected': len(faces),
                'output_image': output_filename,
                'confidence': f"{np.mean(confidence_scores)*100:.1f}%",
                'message': f'Successfully detected {len(faces)} face(s)'
            }
            
        except Exception as e:
            return {'success': False, 'error': f'Processing error: {str(e)}'}
    
    def estimate_age_advanced(self, face_roi, face_width, face_height, img_shape):
        """Advanced age estimation using multiple facial features"""
        try:
            gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            img_height, img_width = img_shape[:2]
            
            # Calculate face size ratio
            face_ratio = (face_width * face_height) / (img_width * img_height)
            
            # Multiple feature analysis
            contrast = np.std(gray_face)
            aspect_ratio = face_width / face_height
            
            # Edge density analysis
            edges = cv2.Canny(gray_face, 50, 150)
            edge_density = np.sum(edges > 0) / (face_width * face_height)
            
            # Skin smoothness analysis
            laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
            
            # Base age estimation with enhanced features
            base_age = 30
            
            # Face size adjustments (more nuanced)
            size_adjustments = [
                (0.15, -15), (0.1, -10), (0.08, -6), (0.06, -3),
                (0.03, +8), (0.02, +12)
            ]
            
            for threshold, adjustment in size_adjustments:
                if face_ratio > threshold:
                    base_age += adjustment
                    break
            
            # Skin texture adjustments
            if contrast < 20: base_age -= 10
            elif contrast < 30: base_age -= 6
            elif contrast > 55: base_age += 12
            elif contrast > 45: base_age += 6
            
            # Face shape adjustments
            if aspect_ratio > 0.9: base_age += 6
            elif aspect_ratio < 0.65: base_age -= 4
            
            # Feature density
            if edge_density > 0.1: base_age += 8
            elif edge_density < 0.02: base_age -= 5
            
            # Skin smoothness (Laplacian variance)
            if laplacian_var < 100: base_age -= 8  # Very smooth skin
            elif laplacian_var > 500: base_age += 10  # Rough skin
            
            # Ensure reasonable age range
            age = max(1, min(100, base_age))
            
            # Confidence calculation
            confidence = 0.5 + min(face_ratio * 8, 0.4)  # Higher confidence for larger faces
            confidence = min(confidence, 0.95)
            
            return age, confidence
            
        except:
            # Fallback estimation
            return 35, 0.5
    
    def draw_face_analysis(self, img, x, y, w, h, age, confidence, face_num):
        """Draw detailed analysis for each face"""
        # Face rectangle
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Age and confidence text
        text = f"Face {face_num}: ~{age}y ({confidence*100:.0f}%)"
        cv2.putText(img, text, (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Face number
        cv2.putText(img, f"#{face_num}", (x, y+h+20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    def draw_final_result(self, img, age, num_faces, avg_confidence):
        """Draw final result on image"""
        # Background for text
        cv2.rectangle(img, (10, 10), (450, 110), (0, 0, 0), -1)
        
        # Main result
        cv2.putText(img, f"Estimated Age: {age} years", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Additional info
        cv2.putText(img, f"Faces detected: {num_faces}", (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(img, f"Confidence: {avg_confidence*100:.1f}%", (20, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        # Success message
        cv2.putText(img, "✅ Face successfully detected!", (img.shape[1]-300, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    def verify_age(self, image_path, output_dir="results"):
        """Age verification - must be 18+"""
        result = self.predict_age(image_path, output_dir)
        
        if result['success']:
            age = result['age']
            is_verified = age >= 18
            
            result['verified'] = is_verified
            result['verification_status'] = "VERIFIED" if is_verified else "NOT VERIFIED"
            result['verification_message'] = f"Age {age} - {'18+' if is_verified else 'Under 18'}"
            
        return result