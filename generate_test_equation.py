from PIL import Image, ImageDraw, ImageFont
import os

# Create directory for test images if it doesn't exist
os.makedirs('test_images', exist_ok=True)

def generate_equation_image(equation, filename, width=500, height=200, bg_color='white', text_color='black'):
    """Generate an image with a mathematical equation."""
    # Create a blank image with white background
    image = Image.new('RGB', (width, height), bg_color)
    draw = ImageDraw.Draw(image)
    
    # Try to use a system font, fallback to default if not available
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 60)
    except IOError:
        font = ImageFont.load_default()
    
    # Calculate text position to center it
    text_width, text_height = draw.textbbox((0, 0), equation, font=font)[2:4]
    position = ((width - text_width) // 2, (height - text_height) // 2)
    
    # Draw the equation
    draw.text(position, equation, font=font, fill=text_color)
    
    # Save the image
    image.save(f'test_images/{filename}')
    print(f"Generated equation image: test_images/{filename}")
    return f'test_images/{filename}'

# Generate some example equations
examples = [
    ("2+2=4", "simple_addition.png"),
    ("5*7=35", "simple_multiplication.png"),
    ("x+5=10", "simple_equation.png"),
    ("3x+2=11", "linear_equation.png"),
    ("25-10=15", "simple_subtraction.png")
]

# Generate all example images
for equation, filename in examples:
    generate_equation_image(equation, filename)

print("All test images generated successfully.")