for (level = 1; level < 7; level++) {
	for (direction = 0; direction < 8; direction++) {
		s_level = toString(level);
		s_direction = toString(direction);
		input_filename = "/home/gulliver/vision-institute/tests/axon-magnification/out/test/guitar_square_band_level_" + s_level + "_bandpass_direction_" + s_direction + "_magnitude.tif";
		open(input_filename);
	    run("Z Project...", "projection=[Max Intensity]");
	    current_window = "MAX_guitar_square_band_level_" + s_level + "_bandpass_direction_" + s_direction + "_magnitude.tif";
	    selectWindow(current_window);
	    run("mpl-viridis");
	    output_filename = "/home/gulliver/vision-institute/tests/axon-magnification/out/test/guitar_square_band_level_" + s_level + "_bandpass_direction_" + s_direction + "_magnitude_max.png";
	    saveAs("PNG", output_filename);
	    close();
	    close();
	}
}