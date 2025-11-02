import os
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import re
import matplotlib.gridspec as gridspec

def analyze_images(image_dir, output_dir):
    """
    Analyzes images in a directory to count white pixels, saves the data to a CSV file,
    and creates a bar plot of the results.

    Args:
        image_dir (str): The path to the directory containing the images.
        output_dir (str): The path to the directory where the output CSV and plot will be saved.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    
    results = []

    for image_file in image_files:
        try:
            with Image.open(os.path.join(image_dir, image_file)) as img:
                # Convert image to numpy array
                img_array = np.array(img.convert('L'))  # Convert to grayscale
                
                # Count white pixels (assuming white is > 200 in grayscale)
                white_pixels = np.sum(img_array > 200)
                total_pixels = img_array.size
                percentage_area = (white_pixels / total_pixels) * 100 if total_pixels > 0 else 0

                # Extract number of dots from filename
                match = re.match(r'(\d+)', image_file)
                num_dots = int(match.group(1)) if match else 0
                
                results.append({
                    'filename': image_file, 
                    'dot_count': num_dots, 
                    'white_pixel_area': white_pixels,
                    'white_pixel_percentage': percentage_area
                })
        except Exception as e:
            print(f"Could not process {image_file}: {e}")

    # Create a DataFrame and save to CSV for absolute values
    df = pd.DataFrame(results)
    df = df.sort_values(by=['dot_count', 'filename']).reset_index(drop=True)
    csv_path_abs = os.path.join(output_dir, 'stimuli_analysis.csv')
    df[['filename', 'dot_count', 'white_pixel_area']].to_csv(csv_path_abs, index=False)
    print(f"Absolute analysis saved to {csv_path_abs}")

    # Save percentage data to a new CSV
    csv_path_perc = os.path.join(output_dir, 'stimuli_analysis_percentage.csv')
    df[['filename', 'dot_count', 'white_pixel_percentage']].to_csv(csv_path_perc, index=False)
    print(f"Percentage analysis saved to {csv_path_perc}")

    # Create and save a plot
    plt.figure(figsize=(15, 8))
    
    # Group by dot_count and calculate the mean white_pixel_area
    grouped_df = df.groupby('dot_count')['white_pixel_area'].mean().reset_index()
    
    plt.bar(grouped_df['dot_count'], grouped_df['white_pixel_area'], color='skyblue')
    plt.title('Average White Pixel Area by Number of Dots')
    plt.xlabel('Number of Dots in Stimulus')
    plt.ylabel('Average White Pixel Area')
    plt.xticks(grouped_df['dot_count'])
    plt.grid(axis='y', linestyle='--')
    
    plot_path = os.path.join(output_dir, 'stimuli_analysis_plot.png')
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    plt.close()

    # Create a composite plot with images below
    fig = plt.figure(figsize=(20, 25)) # A tall figure
    gs_main = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1, 1.5], hspace=0.3)

    # Top: Bar plot
    ax_bar = fig.add_subplot(gs_main[0])
    df['x_label'] = df['filename'].str.replace('.jpg', '')
    ax_bar.bar(df['x_label'], df['white_pixel_area'], color='lightgreen')
    ax_bar.set_title('White Pixel Area for Each Individual Stimulus')
    ax_bar.set_ylabel('White Pixel Area')
    ax_bar.tick_params(axis='x', rotation=90)
    ax_bar.grid(axis='y', linestyle='--')

    # Bottom: Images
    # Add a title for the image section
    ax_images_title = fig.add_subplot(gs_main[1])
    ax_images_title.set_title('Stimulus Images', y=1.0)
    ax_images_title.axis('off')

    gs_images = gridspec.GridSpecFromSubplotSpec(5, 6, subplot_spec=gs_main[1], wspace=0.1, hspace=0.4)

    variants = ['a', 'b', 'c', 'd', 'e']
    ax_6e = None # To store the axis for the '6e' image
    for i in range(1, 7): # numbers 1-6
        for j, var in enumerate(variants): # variants a-e
            img_filename = f'{i}{var}.jpg'
            img_path = os.path.join(image_dir, img_filename)
            try:
                with Image.open(img_path) as img:
                    ax_img = fig.add_subplot(gs_images[j, i-1])
                    ax_img.imshow(img, cmap='gray', vmin=0, vmax=255)
                    ax_img.axis('off')
                    # Add filename label, marking 'e' variants
                    label = img_filename.replace('.jpg','')
                    if 'e' in img_filename:
                        label += '*'
                    ax_img.text(0.5, -0.15, label, size=10, ha="center", transform=ax_img.transAxes)
                    if j == 0: # Add column title
                        ax_img.set_title(f'Number {i}')
                    if img_filename == '6e.jpg':
                        ax_6e = ax_img
            except FileNotFoundError:
                print(f"Image not found: {img_path}")

    plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout, leave space for suptitle
    fig.suptitle('Individual Stimulus Analysis (Absolute)', fontsize=16)
    # Add legend for asterisk under the sixth column
    if ax_6e:
        ax_6e.text(0.5, -0.5, '*stimulus', size=10, ha="center", va="top", transform=ax_6e.transAxes)
 
    individual_plot_path = os.path.join(output_dir, 'stimuli_analysis_individual_plot_absolute.png')
    plt.savefig(individual_plot_path)
    print(f"Individual absolute plot saved to {individual_plot_path}")
    plt.close()

    # Create a composite plot for percentages
    fig_perc = plt.figure(figsize=(20, 25))
    gs_main_perc = gridspec.GridSpec(2, 1, figure=fig_perc, height_ratios=[1, 1.5], hspace=0.3)

    # Top: Bar plot for percentages
    ax_bar_perc = fig_perc.add_subplot(gs_main_perc[0])
    df['x_label'] = df['filename'].str.replace('.jpg', '')
    ax_bar_perc.bar(df['x_label'], df['white_pixel_percentage'], color='lightcoral')
    ax_bar_perc.set_title('White Pixel Area Percentage for Each Individual Stimulus')
    ax_bar_perc.set_ylabel('White Pixel Area (%)')
    ax_bar_perc.tick_params(axis='x', rotation=90)
    ax_bar_perc.grid(axis='y', linestyle='--')

    # Bottom: Images
    ax_images_title_perc = fig_perc.add_subplot(gs_main_perc[1])
    ax_images_title_perc.set_title('Stimulus Images', y=1.0)
    ax_images_title_perc.axis('off')

    gs_images_perc = gridspec.GridSpecFromSubplotSpec(5, 6, subplot_spec=gs_main_perc[1], wspace=0.1, hspace=0.4)

    variants = ['a', 'b', 'c', 'd', 'e']
    ax_6e_perc = None # To store the axis for the '6e' image
    for i in range(1, 7):
        for j, var in enumerate(variants):
            img_filename = f'{i}{var}.jpg'
            img_path = os.path.join(image_dir, img_filename)
            try:
                with Image.open(img_path) as img:
                    ax_img = fig_perc.add_subplot(gs_images_perc[j, i-1])
                    ax_img.imshow(img, cmap='gray', vmin=0, vmax=255)
                    ax_img.axis('off')
                    # Add filename label, marking 'e' variants
                    label = img_filename.replace('.jpg','')
                    if 'e' in img_filename:
                        label += '*'
                    ax_img.text(0.5, -0.15, label, size=10, ha="center", transform=ax_img.transAxes)
                    if j == 0:
                        ax_img.set_title(f'Number {i}')
                    if img_filename == '6e.jpg':
                        ax_6e_perc = ax_img
            except FileNotFoundError:
                print(f"Image not found: {img_path}")
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    fig_perc.suptitle('Individual Stimulus Analysis (Percentage)', fontsize=16)
    # Add legend for asterisk under the sixth column
    if ax_6e_perc:
        ax_6e_perc.text(0.5, -0.5, '*stimulus', size=10, ha="center", va="top", transform=ax_6e_perc.transAxes)

    percentage_plot_path = os.path.join(output_dir, 'stimuli_analysis_individual_plot_percentage.png')
    plt.savefig(percentage_plot_path)
    print(f"Individual percentage plot saved to {percentage_plot_path}")
    plt.close()

if __name__ == '__main__':
    # The path to the stimuli images.
    # The script is in 'stimuli/', so we go into the subdirectory.
    stimuli_path = os.path.join('stimuli', 'Stimuli for EEG-Number2013-20250715T225621Z-1-001', 'Stimuli for EEG-Number2013')
    output_path = os.path.join('stimuli', 'output')
    
    analyze_images(stimuli_path, output_path)
