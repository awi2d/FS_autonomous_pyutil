camL3_bb: 
	camL3_frame_{frnr}.txt contains the bounding boxes for frame frnr of the Left cam during run 3.
	each row is one bounding box, with the format
	{classindex} {width_position} {height_position} {width_size} {height_size}
	classindex: 0: blue, 1: yellow, 2: orange
	position and size in pixel_position/size of the image in that direction.
	So the upper left corner of the bounding box is at pixel ( (width_position-0.5*width_size)*width_of_image, (height_position-0.5*height_size)*height_of_image)
	TODO frame 1306 and 1308 have 14 cones visible, frame 1307 only 7? check.
camL3_poiibb:
	same format as camL3_bb, but the classindex is the point-of-interest-id, instead of color. see ../point_of_interest_numbers.jpg for the meaning of these numbers.

camR3_poiibb:
	same as camL3_poiibb, but for the right camera

bbi_poii:
	each file has the same name as a file in camL3_bb
	the number in a row of bbi_poii is the poii index of the cone annotated in the same row of the file with the same name in bbi_poii.
	This will be removed once all bounding box labels are changed to poii as class instead of color.

cone_annotations.csv:
	each row has the format 
	/cones/camL3_frame_2500.jpg_cone_5.jpg	0.5833333333333334#0.2088888888888889	0.4677777777777778#0.3933333333333333	0.4022222222222222#0.5177777777777778	0.2588888888888889#0.76	0.6522222222222223#0.40444444444444444	0.6855555555555556#0.5511111111111111	0.7133333333333334#0.7633333333333333
	/cones/{camname}_frame_{framenumber}.jpg_cone_{conenumber}.jpg,{width_position}#{height_position},{width_position}#{height_position},{width_position}#{height_position},{width_position}#{height_position},{width_position}#{height_position},{width_position}#{height_position},{width_position}#{height_position}
	camname: camL3 is the left cam during run 3, camR3 is the right cam during run 3.
	the keypoints are always in the order shown in ../cone_keypoints.jpg
	conenumber is the row in {camname}_bb/{camname}_frame_{framenumber}.txt that contains the bounding box for this cone.
	currently only cones of camL3 are annotated. when poiibb is annotated, conenumber might change to class instead of row.

droneview_annotations.csv
	each row has the format
	/droneview/drone3_frame_2366.jpg,{width_position}#{height_position},... (positions for all 88 poi)
	width and height position are relative to image size.