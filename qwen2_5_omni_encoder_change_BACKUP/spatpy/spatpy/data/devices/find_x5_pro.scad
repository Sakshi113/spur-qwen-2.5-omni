body_dimensions = [163.6, 8.26, 74.0];
body_curve_radius = 1.5;
body_curve_diameter = 2 * body_curve_radius;
bezel_start_size = 40;
bezel_end_size = 25;
bezel_curve_radius = 4;
bezel_height = 2.3;


$fn=32;

translate([0, 0, body_dimensions.z])
rotate([0, 180, 90])
translate([body_curve_radius, body_curve_radius, body_curve_radius]) {
    minkowski() {
        union() {
            translate([
                bezel_start_size / 2 + bezel_curve_radius,
                0,
                bezel_start_size / 2 + bezel_curve_radius
            ]) {
                rotate([90, 0, 0]) {
                    linear_extrude(
                        height=bezel_height,
                        scale=(bezel_end_size/bezel_start_size)
                    ) {
                        offset(r=bezel_curve_radius) {
                            square(bezel_start_size, center=true);
                        }
                    }
                }
            }
            cube([
                body_dimensions.x - body_curve_diameter,
                body_dimensions.y - body_curve_diameter,
                body_dimensions.z - body_curve_diameter
            ]);                
         }
         sphere(d=body_curve_diameter);
     }
 }