import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Tonnage Predictor")

# Function to process image
def process_image(image, tons_per_in_sq, num_cavities):
    # Initialize pixel_per_cm
    pixel_per_cm = None

    # Load YOLOv5 model
    model = YOLO("yolov8m-seg-custom.pt")

    # Detect objects using YOLOv5
    results = model.predict(source=image, show=False)

    for result in results:
        # Get bounding box coordinates for each image
        bounding_boxes = result.boxes.xyxy  # Access bounding box coordinates in [x1, y1, x2, y2] format

        # Draw circles around the detected objects
        for box in bounding_boxes:
            x1, y1, x2, y2 = box[:4].int().tolist()  # Convert tensor to list
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            radius = max(abs(x2 - x1), abs(y2 - y1)) // 2
            cv2.circle(image, (center_x, center_y), radius, (0, 255, 0), 2)  # Draw circle
            
            # Calculate dimensions of the reference object (coin)
            ref_w, ref_h = abs(x2 - x1), abs(y2 - y1)
            dist_in_pixel = max(ref_w, ref_h)  # Assuming the longer side of the bounding box as the reference size
            
            # Diameter of the coin in cm
            ref_coin_diameter_cm = 2.38
            
            # Calculate pixel-to-cm conversion factor
            pixel_per_cm = dist_in_pixel / ref_coin_diameter_cm

    # Check if no objects are detected
    if pixel_per_cm is None:
        st.error("No reference objects detected in the image. Please recapture.")
        return None

    # Find contours
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    edged = cv2.Canny(blur, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # Filter out contours detected by YOLO
    filtered_contours = []
    for cnt in cnts:
        if cv2.contourArea(cnt) > 50:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            # Check if the contour falls within any bounding box of objects detected by YOLO
            contour_in_yolo_object = False
            for yolo_box in bounding_boxes:
                yolo_x1, yolo_y1, yolo_x2, yolo_y2 = yolo_box[:4].int().tolist()
                if yolo_x1 < rect[0][0] < yolo_x2 and yolo_y1 < rect[0][1] < yolo_y2:
                    contour_in_yolo_object = True
                    break

            if not contour_in_yolo_object:
                filtered_contours.append(cnt)

    # Find the contour with the largest area
    largest_contour = max(filtered_contours, key=cv2.contourArea)

    # Draw contour of the object with the largest area
    if largest_contour is not None:
        # Draw contour of the object
        cv2.drawContours(image, [largest_contour], -1, (0, 0, 255), 2)  # Draw contour line instead of bounding box
                
        # Calculate dimensions and area of the object
        area = cv2.contourArea(largest_contour)
        area_cm2 = abs(area) / (pixel_per_cm ** 2)
        # Calculate dimensions of the object in centimeters
        rect = cv2.minAreaRect(largest_contour)
        (x, y), (width_px, height_px), angle = rect
        width_cm = width_px / pixel_per_cm
        height_cm = height_px / pixel_per_cm

    area_in2 = area_cm2 / 2.54**2
    # If area is less than 1, check shape of contour line and calculate area accordingly
    if area_in2 < 1:
            area_in2 = width_cm * height_cm / 2.54 ** 2

    # Calculate tonnage
    tonnage = calculate_tonnage(area_in2, tons_per_in_sq, num_cavities)

    st.markdown(f'<div style="background-color: blue; color: white; padding: 20px; font-size: 36px; font-weight: bold;">Predicted Tonnage is: {round(tonnage, 2)} tons</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="color: blue; padding: 20px; font-size: 24px; font-weight: bold;">Calculated Area is: {round(area_in2, 2)} square inches</div>', unsafe_allow_html=True)

    return image


# Function to calculate tonnage based on area
def calculate_tonnage(area_in2, tons_per_in_sq, num_cavities):
    # Calculate tonnage
    tonnage = area_in2 * num_cavities * tons_per_in_sq
    return tonnage


# Display logo
logo_image = Image.open("FIMMTech-logo.png")
st.image(logo_image, use_column_width=True)

st.markdown("---")
# Main Streamlit app with centered title
st.markdown("<h1 style='text-align: center;'>Object Tonnage Predictor</h1>", unsafe_allow_html=True)
# st.markdown("---")
st.markdown("<br>", unsafe_allow_html=True)


# Divide the layout into two columns with a different ratio
col1, col2 = st.columns([2, 2])
with col1:
    st.markdown("<h4 style='text-align: center;'>Upload Image</h4>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=['jpg', 'png', 'jpeg'])
    # Input fields on one side
    if uploaded_file is not None:
        # Read the uploaded file bytes
        file_bytes = uploaded_file.read()

        # Decode the bytes using OpenCV
        image = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), 1)

        # Check if the image is decoded successfully
        if image is not None:
            # Display the image
            st.image(image, channels="BGR", caption="Uploaded Image", use_column_width=True)
        else:
            st.error("Failed to decode the image reupload!")

    
# Input fields in col2
with col2:
    num_cavities = 0
    st.markdown("<h4 style='text-align: center;'>Number of Cavities</h4>", unsafe_allow_html=True)
    num_cavities = st.number_input("Number of Cavities", key="num_cavities", value=int(num_cavities), step=1)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<h4 style='text-align: center;'>Tons per Inch Square</h4>", unsafe_allow_html=True)
    tons_per_in_sq = st.number_input("", key="tons_per_in_sq")


    # Calculate Tonnage button
    calculate_button = st.button("Calculate Tonnage")

    
# Styling for the button
button_style = """
    <style>
    .stButton>button {
        background-color: brown;
        color: white;
        padding: 10px 24px;
        border: none;
        border-radius: 4px;
        cursor: pointer;
    }
    </style>
"""

# Display the button with custom styling
st.markdown(button_style, unsafe_allow_html=True)

# Add space between logo and main content
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")


# Process and display the image outside the columns
if uploaded_file is not None and calculate_button:
    # Process the image
    processed_image = process_image(image, num_cavities, tons_per_in_sq)
    if processed_image is not None:
        # Display the processed image
        st.image(processed_image, channels="BGR", caption="Detected Objects", use_column_width=True)

elif calculate_button:
    st.error("No image uploaded! Please upload")

