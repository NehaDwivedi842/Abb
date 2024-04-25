# import cv2
# import numpy as np
# import streamlit as st
# import imutils
# from PIL import Image
# from ultralytics import YOLO

# st.title("Object Tonnage Predictor")

# # Function to process image
# def process_image(image, tons_per_in_sq, num_cavities):
#     # Load YOLOv5 model
#     model = YOLO("yolov8m-seg-custom.pt")

#     # Detect objects using YOLOv5
#     results = model.predict(source=image, show=False)

#     for result in results:
#         # Get bounding box coordinates for each image
#         bounding_boxes = result.boxes.xyxy  # Access bounding box coordinates in [x1, y1, x2, y2] format

#         # Draw circles around the detected objects
#         for box in bounding_boxes:
#             x1, y1, x2, y2 = box[:4].int().tolist()  # Convert tensor to list
#             center_x = int((x1 + x2) / 2)
#             center_y = int((y1 + y2) / 2)
#             radius = max(abs(x2 - x1), abs(y2 - y1)) // 2
#             cv2.circle(image, (center_x, center_y), radius, (0, 255, 0), 2)  # Draw circle
            
#             # Calculate dimensions of the reference object (coin)
#             ref_w, ref_h = abs(x2 - x1), abs(y2 - y1)
#             dist_in_pixel = max(ref_w, ref_h)  # Assuming the longer side of the bounding box as the reference size
            
#             # Diameter of the coin in cm
#             ref_coin_diameter_cm = 2.426
            
#             # Calculate pixel-to-cm conversion factor
#             pixel_per_cm = dist_in_pixel / ref_coin_diameter_cm

#             # Draw reference object size message above the detected object
#             ref_text = "Reference object size=0.955"
#             cv2.putText(image, ref_text, (center_x - 150, center_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

#     if pixel_per_cm is None:
#         print("No objects detected in the image. Please recapture.")

#     # Find contours
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (9, 9), 0)
#     edged = cv2.Canny(blur, 50, 100)
#     edged = cv2.dilate(edged, None, iterations=1)
#     edged = cv2.erode(edged, None, iterations=1)
#     cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     cnts = imutils.grab_contours(cnts)

#     # Filter out contours detected by YOLO
#     filtered_contours = []
#     for cnt in cnts:
#         if cv2.contourArea(cnt) > 50:
#             rect = cv2.minAreaRect(cnt)
#             box = cv2.boxPoints(rect)
#             box = np.int0(box)
            
#             # Check if the contour falls within any bounding box of objects detected by YOLO
#             contour_in_yolo_object = False
#             for yolo_box in bounding_boxes:
#                 yolo_x1, yolo_y1, yolo_x2, yolo_y2 = yolo_box[:4].int().tolist()
#                 if yolo_x1 < rect[0][0] < yolo_x2 and yolo_y1 < rect[0][1] < yolo_y2:
#                     contour_in_yolo_object = True
#                     break

#             if not contour_in_yolo_object:
#                 filtered_contours.append(cnt)

#     # Find the contour with the largest area
#     largest_contour = max(filtered_contours, key=cv2.contourArea)

#     # Draw contour of the object with the largest area
#     if largest_contour is not None:
#         # Draw contour of the object
#         cv2.drawContours(image, [largest_contour], -1, (0, 0, 255), 2)  # Draw contour line instead of bounding box
        
#         # Calculate dimensions and area of the object
#         area_cm2 = cv2.contourArea(largest_contour) / (pixel_per_cm ** 2)
#         rect = cv2.minAreaRect(largest_contour)
#         (x, y), (width_px, height_px), angle = rect
#         width_cm = width_px / pixel_per_cm
#         height_cm = height_px / pixel_per_cm

#     # If area is less than 1, check shape of contour line and calculate area accordingly
#     if area_cm2 < 1:
#         # Calculate aspect ratio of the bounding rectangle
#         aspect_ratio = width_px / height_px

#         # If aspect ratio is less than 1, consider it as a long and narrow shape (e.g., rectangle)
#         if aspect_ratio < 1:
#             area_cm2 = np.pi * ((width_cm / 2) ** 2)

#     # Calculate text positions
#     text_x = int(x - 100)
#     text_y = int(y - 20)

#     # Calculate dimensions and area of the object in inches
#     width_in = width_cm / 2.54
#     height_in = height_cm / 2.54
#     area_in2 = area_cm2 / 2.54

#     # Calculate tonnage
#     tonnage = calculate_tonnage(area_in2, tons_per_in_sq, num_cavities)

#     # Draw text annotations with dimensions in inches and tonnage
#     cv2.putText(image, "Length: {:.1f}in".format(width_in), (text_x, text_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
#     cv2.putText(image, "Breadth: {:.1f}in".format(height_in), (text_x, text_y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
#     cv2.putText(image, "Area: {:.1f}in^2".format(area_in2), (text_x, text_y + 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
#     cv2.putText(image, "Tonnage: {:.2f}".format(tonnage), (text_x, text_y + 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

#     # Display the size above the image
#     cv2.putText(image, "Coin is the reference Object", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 1), 2)
#     st.success(f" Length: {height_in:.1f}in, Breadth: {width_in:.1f}in, Projected Area: {area_in2:.1f}in^2, Tonnage: {tonnage:.2f}")

#     return image

# # Function to calculate tonnage based on area
# def calculate_tonnage(area_in2, tons_per_in_sq, num_cavities):
#     # Calculate tonnage
#     tonnage = area_in2 * num_cavities * tons_per_in_sq
#     return tonnage

# # Main Streamlit app
# uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
# num_cavities = st.number_input("Number of cavities:")
# tons_per_in_sq = st.number_input("Tons per inch square:")

# if uploaded_file is not None:
#     # Convert the file to an opencv image
#     file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#     image = cv2.imdecode(file_bytes, 1)

#     # Process and display the image
#     if st.button("Calculate Tonnage"):
#         processed_image = process_image(image, num_cavities, tons_per_in_sq)
#         if processed_image is not None:
#             st.image(processed_image, channels="BGR", caption="Detected Objects", use_column_width=True)

import cv2
import numpy as np
import streamlit as st
import imutils
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Object Tonnage Predictor", page_icon="📏")

# Function to process image
def process_image(image, tons_per_in_sq, num_cavities):
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
            ref_coin_diameter_cm = 2.426
            
            # Calculate pixel-to-cm conversion factor
            pixel_per_cm = dist_in_pixel / ref_coin_diameter_cm

            # Draw reference object size message above the detected object
            ref_text = "Reference object size=0.955"
            cv2.putText(image, ref_text, (center_x - 150, center_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    if pixel_per_cm is None:
        print("No objects detected in the image. Please recapture.")

    # Find contours
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    edged = cv2.Canny(blur, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

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
        area_cm2 = cv2.contourArea(largest_contour) / (pixel_per_cm ** 2)
        rect = cv2.minAreaRect(largest_contour)
        (x, y), (width_px, height_px), angle = rect
        width_cm = width_px / pixel_per_cm
        height_cm = height_px / pixel_per_cm

    # If area is less than 1, check shape of contour line and calculate area accordingly
    if area_cm2 < 1:
        # Calculate aspect ratio of the bounding rectangle
        aspect_ratio = width_px / height_px

        # If aspect ratio is less than 1, consider it as a long and narrow shape (e.g., rectangle)
        if aspect_ratio < 1:
            area_cm2 = np.pi * ((width_cm / 2) ** 2)

    # Calculate text positions
    text_x = int(x - 100)
    text_y = int(y - 20)

    # Calculate dimensions and area of the object in inches
    width_in = width_cm / 2.54
    height_in = height_cm / 2.54
    area_in2 = area_cm2 / 2.54

    # Calculate tonnage
    tonnage = calculate_tonnage(area_in2, tons_per_in_sq, num_cavities)

    # Draw text annotations with dimensions in inches and tonnage
    cv2.putText(image, "Length: {:.1f}in".format(width_in), (text_x, text_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(image, "Breadth: {:.1f}in".format(height_in), (text_x, text_y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(image, "Area: {:.1f}in^2".format(area_in2), (text_x, text_y + 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(image, "Tonnage: {:.2f}".format(tonnage), (text_x, text_y + 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Display the size above the image
    cv2.putText(image, "Coin is the reference Object", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 1), 2)
    # Display success message with predicted tonnage
    st.success(f"##### Predicted Tonnage is: {tonnage}")


    return image

# Function to calculate tonnage based on area
def calculate_tonnage(area_in2, tons_per_in_sq, num_cavities):
    # Calculate tonnage
    tonnage = area_in2 * num_cavities * tons_per_in_sq
    return tonnage


    # Display logo
logo_image = Image.open("LogoHeader-600x85.png")
st.image(logo_image, use_column_width=True)

st.markdown("---")
# Main Streamlit app with centered title
st.markdown("<h1 style='text-align: center;'>Object Tonnage Predictor</h1>", unsafe_allow_html=True)
# st.markdown("---")
st.markdown("<br>", unsafe_allow_html=True)


# Divide the layout into two columns with a different ratio
col1, col2 = st.columns([2, 2])

# Input fields on one side


# Input fields in col1
with col1:
    st.markdown("<h4 style='text-align: center;'>Upload Image</h4>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=['jpg', 'png', 'jpeg'])

# Input fields in col2
with col2:
    st.markdown("<h4 style='text-align: center;'>Number of Cavities</h4>", unsafe_allow_html=True)
    num_cavities = st.number_input("", key="num_cavities")

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
    # Convert the file to an opencv image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Process the image
    processed_image = process_image(image, num_cavities, tons_per_in_sq)
    if processed_image is not None:
        # Display the processed image
        st.image(processed_image, channels="BGR", caption="Detected Objects", use_column_width=True)


elif calculate_button:
    st.error("No image uploaded! Please upload")
