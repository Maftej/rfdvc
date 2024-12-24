import os
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.interpolate import make_interp_spline
from matplotlib.ticker import MultipleLocator


class PacketLossPlotManager:
    def __init__(self):
        pass

    def plot_packet_loss(self, command_data):
        # Retrieve paths from command_data
        plots_path = command_data.get("plots_path", "")
        pl_dataframe_paths = command_data.get("pl_dataframe_paths", [])

        # Ensure the plots directory exists
        if not os.path.exists(plots_path):
            os.makedirs(plots_path)

        # Initialize an empty list to store DataFrames
        dataframes = []

        # Load each DataFrame and append it to the list
        for path in pl_dataframe_paths:
            try:
                df = pd.read_csv(path)
                dataframes.append(df)
                print(f"DataFrame loaded successfully from {path}")
            except Exception as e:
                print(f"Error loading DataFrame from {path}: {e}")

        # Combine all DataFrames into a single DataFrame
        if dataframes:
            df = pd.concat(dataframes, ignore_index=True)
        else:
            print("No DataFrames were loaded. Exiting the plotting function.")
            return

        # Convert file sizes to kilobytes for readability
        df['Original_File_Size_KB'] = df['Original_File_Size_Bytes'] / 1024
        df['Corrupted_File_Size_KB'] = df['Corrupted_File_Size_Bytes'] / 1024

        # Define the resolutions and codecs of interest
        resolution_order = ['500x280', '720x404', '1920x1080']
        codec_order = ['h264', 'h265']

        # Filter the DataFrame to include only the specified resolutions
        df = df[df['Resolution'].isin(resolution_order)]

        # Ensure Codec and Approach columns are lowercase
        df['Codec'] = df['Codec'].str.lower()
        df['Approach'] = df['Approach'].str.lower()



        # Plot BLER vs SSIM
        self.plot_bler_vs_ssim(name="trends", df=df, plots_path=plots_path, max_count=10)
        self.count_approach_files(dataframe=df)
        # Plot by weather
        # self.plot_bler_vs_ssim_by_weather(df, plots_path)

    def count_approach_files(self, dataframe):
        """
        Count the number of 'rfdvc' and 'vc' files for h264 and h265 codecs in the provided DataFrame,
        grouped by weather and resolution.

        Args:
            dataframe (pd.DataFrame): The DataFrame containing 'Approach', 'Codec', 'Resolution', and 'Weather' columns.

        Returns:
            dict: A dictionary with counts of 'rfdvc' and 'vc' approaches for h264 and h265, grouped by weather and resolution.
        """
        # Ensure the necessary columns exist
        required_columns = ['Approach', 'Codec', 'Resolution', 'Weather']
        if not all(col in dataframe.columns for col in required_columns):
            print(f"Missing required columns in DataFrame: {', '.join(required_columns)}")
            return {}

        # Ensure the columns are in lowercase
        dataframe['Approach'] = dataframe['Approach'].str.lower()
        dataframe['Codec'] = dataframe['Codec'].str.lower()
        dataframe['Resolution'] = dataframe['Resolution'].str.lower()
        dataframe['Weather'] = dataframe['Weather'].str.lower()

        # Initialize a dictionary to store counts grouped by weather and resolution
        counts = {}

        # Group by Weather and Resolution, then count occurrences of each codec and approach combination
        for weather in dataframe['Weather'].unique():
            weather_df = dataframe[dataframe['Weather'] == weather]

            for resolution in weather_df['Resolution'].unique():
                resolution_df = weather_df[weather_df['Resolution'] == resolution]

                # Count for h264 rfdvc
                h264_rfdvc_count = len(resolution_df[
                                           (resolution_df['Codec'] == 'h264') & (resolution_df['Approach'] == 'rfdvc')
                                           ])

                # Count for h265 rfdvc
                h265_rfdvc_count = len(resolution_df[
                                           (resolution_df['Codec'] == 'h265') & (resolution_df['Approach'] == 'rfdvc')
                                           ])

                # Count for h264 vc
                h264_vc_count = len(resolution_df[
                                        (resolution_df['Codec'] == 'h264') & (resolution_df['Approach'] == 'vc')
                                        ])

                # Count for h265 vc
                h265_vc_count = len(resolution_df[
                                        (resolution_df['Codec'] == 'h265') & (resolution_df['Approach'] == 'vc')
                                        ])

                # Add to the counts dictionary
                counts[(weather, resolution)] = {
                    'h264_rfdvc': h264_rfdvc_count,
                    'h265_rfdvc': h265_rfdvc_count,
                    'h264_vc': h264_vc_count,
                    'h265_vc': h265_vc_count
                }

        # Print the counts for verification
        for (weather, resolution), resolution_counts in counts.items():
            print(f"Weather: {weather}, Resolution: {resolution}")
            print(f"  H264 RFDVC Count: {resolution_counts['h264_rfdvc']}")
            print(f"  H265 RFDVC Count: {resolution_counts['h265_rfdvc']}")
            print(f"  H264 VC Count: {resolution_counts['h264_vc']}")
            print(f"  H265 VC Count: {resolution_counts['h265_vc']}")

    def plot_bler_vs_ssim_by_weather(self, df, plots_path, max_count=10):
        """
        Filters data by weather condition and uses the existing plot_bler_vs_ssim function to generate plots.
        """
        filter_type = "Weather"

        # Ensure 'Weather' column exists
        if filter_type not in df.columns:
            print("The column 'Weather' is missing from the DataFrame.")
            return

        # Get unique weather conditions
        weather_conditions = df[filter_type].unique()

        for weather in weather_conditions:
            # Filter data for the current weather
            weather_df = df[df[filter_type] == weather]

            if weather_df.empty:
                print(f"No data available for weather condition: {weather}")
                continue

            # Print a message to indicate the current weather being plotted
            print(f"Plotting data for weather condition: {weather}")

            # Call the existing plot function with the filtered DataFrame
            print(fr"{filter_type}=", weather)
            self.plot_bler_vs_ssim(name=weather, df=weather_df, plots_path=plots_path, max_count=max_count)

    def plot_bler_vs_ssim(self, name="trends", df=None, plots_path=None, max_count=10):
        # Filter out necessary columns for BLER vs SSIM plotting
        if 'BLER' not in df.columns or 'SSIM' not in df.columns:
            print("Required columns (BLER, SSIM) are missing from the DataFrame.")
            return

        # Filter valid BLER and SSIM values
        df_filtered = df[(df['BLER'] >= 0) & (df['SSIM'] >= 0) & (df['SSIM'] <= 1)].copy()

        if df_filtered.empty:
            print("No data available after filtering for BLER vs SSIM plot.")
            return

        # Calculate the dynamic bin size based on the range of BLER values and the desired max_count
        bler_min = df_filtered['BLER'].min()
        bler_max = df_filtered['BLER'].max()
        bler_range = bler_max - bler_min
        bin_size = bler_range / max_count  # Dynamic bin size to limit the number of points on the x-axis

        # Define bins for BLER and group data by these bins
        df_filtered['BLER_Bin'] = (df_filtered['BLER'] // bin_size) * bin_size  # Create bins based on dynamic bin size

        # Group by BLER bins, Codec, and Approach; calculate the average SSIM
        df_grouped = df_filtered.groupby(['Codec', 'Approach', 'BLER_Bin']).agg(
            Avg_SSIM=('SSIM', 'mean'),
            Avg_BLER=('BLER', 'mean')
        ).reset_index()

        # Define the color scheme for the different codec + approach combinations
        combined_palette = {
            'h264 rfdvc': '#0057b7',  # Blue
            'h264 vc': '#4b4b4b',     # Dark Gray
            'h265 rfdvc': '#ffd800',  # Yellow
            'h265 vc': '#d0d0d0'      # Light Gray
        }

        # Prepare the plot
        plt.figure(figsize=(10, 6))

        # Plot data for each combination (RFDVC, VC, h264, h265)
        for codec in ['h264', 'h265']:
            for approach in ['rfdvc', 'vc']:
                # Filter the data for this codec and approach
                filtered_df = df_grouped[(df_grouped['Codec'] == codec) & (df_grouped['Approach'] == approach)]

                if filtered_df.empty:
                    print(f"No data available for {codec.upper()} {approach.upper()}.")
                    continue

                # Sort by BLER for correct plotting order
                filtered_df = filtered_df.sort_values(by='Avg_BLER')

                # Create label and plot the line
                # Create label and plot the line
                codec_label = "H.264" if codec == 'h264' else "H.265"
                label = f"{codec_label} {approach.upper()}"
                palette_key = f"{codec} {approach}"
                color = combined_palette.get(palette_key, '#000000')  # Default to black if key not found
                plt.plot(filtered_df['Avg_BLER'], filtered_df['Avg_SSIM'], label=label,
                         color=color, marker='o', linestyle='-', linewidth=2.8, markersize=10)

                print("*"*40)
                print(f"CODEC={codec}")
                print(f"APPROACH={approach}")
                print(filtered_df['Avg_BLER'])
                print(filtered_df['Avg_SSIM'])
                print("*"*40)


        # Set axis labels and title
        plt.title('Comparison of Packet Loss between RFDVC and VC', fontsize=19)
        plt.xlabel('BLER', fontsize=18)
        plt.ylabel('SSIM', fontsize=18)

        # x=0.8, y=0.9
        x_lim = 0.8
        y_lim = 0.9
        # Set custom axis limits
        plt.xlim(0, x_lim)  # Limit x-axis to 0 - 0.8
        plt.ylim(0, y_lim)  # Limit y-axis to 0 - 0.9

        # Ensure ticks from 0.1 to 1.0 are explicitly shown
        x_ticks = np.arange(0.1, (x_lim + 0.1), 0.1)  # Generate ticks from 0.1 to 0.8
        y_ticks = np.arange(0.1, (y_lim + 0.1), 0.1)  # Generate ticks from 0.1 to 0.9

        plt.gca().set_xticks(x_ticks)  # Set X-axis ticks
        plt.gca().set_yticks(y_ticks)  # Set Y-axis ticks

        plt.yticks(fontsize=16)
        plt.xticks(fontsize=16)

        plt.legend(title='', loc='lower left', fontsize=16)
        plt.tight_layout()
        plt.grid(True)

        # Get timestamp for filename
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        # Save the plot
        bler_ssim_plot_path_jpg = os.path.join(plots_path, f'bler_vs_ssim_{name}_{timestamp}.jpg')
        bler_ssim_plot_path_pdf = os.path.join(plots_path, f'bler_vs_ssim_{name}_{timestamp}.pdf')
        plt.savefig(bler_ssim_plot_path_jpg, format='jpg')
        plt.savefig(bler_ssim_plot_path_pdf, format='pdf')
        plt.close()
        print(f"BLER vs SSIM plot (trends) saved as {bler_ssim_plot_path_jpg} and {bler_ssim_plot_path_pdf}")

        # Group by BLER bins, Codec, and Approach; calculate the average SSIM and count the number of values
        df_grouped = df_filtered.groupby(['Codec', 'Approach', 'BLER_Bin']).agg(
            Avg_SSIM=('SSIM', 'mean'),
            Avg_BLER=('BLER', 'mean'),
            Count=('SSIM', 'count')  # Count the number of values for each BLER bin
        ).reset_index()

        # Print the number of values averaged for each BLER bin
        for _, row in df_grouped.iterrows():
            print(
                f"Codec: {row['Codec']}, Approach: {row['Approach']}, BLER Bin: {row['BLER_Bin']}, Count: {row['Count']}")

