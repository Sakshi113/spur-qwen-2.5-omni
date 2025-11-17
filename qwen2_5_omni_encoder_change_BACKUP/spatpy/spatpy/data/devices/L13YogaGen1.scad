/*
NOTE: 
- Measurements were made rounded to nearest mm with enforced symmetry. As the microphone spacing measurements do not necessarily add up to the overall width measurements. Microphones are usually <1mm thick so the outermost microphone spacing measurements can be adjusted to match the desired width if necessary
- 'x' represents microphones
- 'wide' is the width for both screen and keyboard panels
- 'screen height' is the distance from top of screen to hinge. Some devices have additional panel past the hinge, but this is behind the KB so can probably be ignored.
- 'KB deep' is distance from front of device (trackpad end) to the hinge
- 'KB thick' is distance from desk to top (keyboard side) of the thickest part of KB section
- Therefore actual microphone position relative to the desk would be veritical distance based on screen height to hinge based on the screen angle, plus the thickness of the the KB.
*/

/*
L13YogaGen1

312 wide

Mics 3mm from top
|--128--x--54--x--128--|

Screen height: 216 to hinge

KB
215 deep
11mm thick
*/

body_width = 312;
kb_depth = 215;
kb_thickness = 11;

mic_distance_from_top = 3;
screen_thickness = 3;
screen_height = 216;
screen_angle = 50;
translate([(screen_height + kb_thickness - mic_distance_from_top) * cos(screen_angle), 0, -(screen_height + kb_thickness - mic_distance_from_top) * sin(screen_angle)]) {
rotate([0, screen_angle - 90, 0]) {
translate([-screen_thickness/2, 0, 0]) {
    linear_extrude(height=(screen_height + kb_thickness)) {
        square([screen_thickness, body_width], center=true);
    }
}
}

rotate([0, 0, 0]) {
    translate([kb_depth/2 - screen_thickness, 0, 0]) {
        linear_extrude(height=kb_thickness) {
            square([kb_depth, body_width], center=true);
        }
    }
}
}
//

//rotate([0, screen_angle - 90, 0]) {
//    
//}
