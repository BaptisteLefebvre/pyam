for (level = 1; level < 7; level++) {
	for (direction = 0; direction < 8; direction++) {
		s_level = toString(level);
		s_direction = toString(direction);
		input_filename = "/home/gulliver/vision-institute/tests/axon-magnification/out/test/guitar_square_band_level_" + s_level + "_bandpass_direction_" + s_direction + "_phase.tif";
		open(input_filename);
	    run("Z Project...", "projection=[Standard Deviation]");
	    current_window = "STD_guitar_square_band_level_" + s_level + "_bandpass_direction_" + s_direction + "_phase.tif";
	    selectWindow(current_window);
	    run("mpl-viridis");
	    output_filename = "/home/gulliver/vision-institute/tests/axon-magnification/out/test/guitar_square_band_level_" + s_level + "_bandpass_direction_" + s_direction + "_phase_std.png";
	    saveAs("PNG", output_filename);
	    close();
	    close();
	}
}