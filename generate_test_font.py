from PIL import Image, ImageDraw, ImageFont
import os

def ttf_to_individual_pngs(text, ttf_path, output_folder, image_size=(200, 100), font_size=30, font_color=(0, 0, 0)):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Load the TrueType Font
    font = ImageFont.truetype(ttf_path, font_size)

    # Get the font name from the ttf_path
    font_name = os.path.splitext(os.path.basename(ttf_path))[0].replace("-","_")

    for char in text:
        # Create a blank image for each character
        image = Image.new("RGBA", image_size, (0, 0, 0, 255))

        # Create a drawing object
        draw = ImageDraw.Draw(image)

        # Calculate the position to center the text
        text_bbox = draw.textbbox((0, 0), char, font=font)
        x = (image_size[0] - text_bbox[2]) // 2
        y = (image_size[1] - text_bbox[3]) // 2

        # Draw text on the image
        draw.text((x, y), char, font=font, fill=font_color)

        # Save the image as PNG with the font name and the character in the filename
        png_path = os.path.join(output_folder, f"{font_name}_{char.lower()}.png")
        image.save(png_path, "PNG")

# Example usage
text_to_convert = "1234567890hkc"
output_folder = "output_images"
image_size = (30, 30)
font_size = 30
font_color = (255, 255, 255)

ttf_file_path = [
    r"C:\Users\Admin\Documents\FinDig\fonts\calibri-bold-italic.ttf",
    r"C:\Users\Admin\Documents\FinDig\fonts\calibri-bold.ttf",
    r"C:\Users\Admin\Documents\FinDig\fonts\calibri-italic.ttf",
    r"C:\Users\Admin\Documents\FinDig\fonts\calibri-regular.ttf"
]
for file in ttf_file_path:
    ttf_to_individual_pngs(text_to_convert, file, output_folder, image_size, font_size, font_color)