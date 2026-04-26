reinitialize
set_color mf_backbone, [0.150, 0.200, 0.280]
set_color mf_backbone_soft, [0.570, 0.630, 0.710]
set_color mf_context_line, [0.720, 0.780, 0.850]
set_color mf_mut, [0.860, 0.360, 0.190]
set_color mf_mut_alt, [0.960, 0.660, 0.180]
set_color mf_design, [0.950, 0.740, 0.220]
set_color mf_protected, [0.040, 0.500, 0.530]
set_color mf_context, [0.250, 0.580, 0.840]
set_color mf_label, [0.100, 0.140, 0.210]
set_color mf_surface, [0.800, 0.880, 0.960]
load F:/4-15Marsprotein/mars_stack/inputs/calb_1lbt/1LBT.pdb, target
remove not (target and chain A)
remove solvent
hide everything, all
bg_color white
set ray_opaque_background, off
set depth_cue, 0
set orthoscopic, on
set antialias, 2
set specular, 0.15
set ambient, 0.50
set direct, 0.18
set ray_trace_mode, 1
set ray_trace_gain, 0.08
set cartoon_fancy_helices, 1
set cartoon_smooth_loops, 1
set cartoon_transparency, 0.00
set stick_radius, 0.16
set sphere_scale, 0.30
set dash_gap, 0.28
set label_font_id, 7
set label_size, -0.45
set label_color, mf_label
show cartoon, target
color mf_backbone, target
select design_resi, target and chain A and resi 249+251+298
select protected_resi, target and chain A and resi 22+64+74+105+187+216+224+258+293+311
select overall_resi, target and chain A and resi 249+251+298
select learned_resi, target and chain A and resi 249+251+298
show lines, protected_resi
color mf_protected, protected_resi
show spheres, design_resi
color mf_design, design_resi
show sticks, overall_resi
show spheres, overall_resi
color mf_mut, overall_resi
show sticks, learned_resi and not overall_resi
color mf_mut_alt, learned_resi and not overall_resi
select design_window, byres (target within 8 of design_resi)
show cartoon, design_window
show lines, design_window and not (overall_resi or learned_resi or protected_resi or design_resi)
color mf_context_line, design_window and not (overall_resi or learned_resi or protected_resi or design_resi)
color mf_backbone_soft, design_window and not (overall_resi or learned_resi or protected_resi)
set cartoon_transparency, 0.10, design_window
set sphere_transparency, 0.08, overall_resi
set sphere_transparency, 0.20, design_resi and not overall_resi
label overall_resi and name CA, resn + resi
disable learned_resi
pseudoatom title_anchor, pos=[0,0,0], label="1LBT"
hide everything, title_anchor
delete title_anchor

# overview scene
hide labels, all
hide lines, design_window and not (overall_resi or protected_resi)
hide surface, all
hide spheres, design_resi and not overall_resi
set sphere_scale, 0.26, overall_resi
set stick_radius, 0.14, overall_resi
set line_width, 1.6, protected_resi
orient target
turn y, -10
turn x, 8
zoom target, 10
scene overview, store
png F:/4-15Marsprotein/mars_stack/outputs/paper_bundle_v1/structure_panels/1LBT/overview.png, width=2400, height=1800, dpi=300, ray=1

# design window scene
disable target
enable target
show cartoon, target
hide labels, all
label overall_resi and name CA, resn + resi
show spheres, design_resi
show sticks, overall_resi
show lines, design_window and not (overall_resi or learned_resi or protected_resi or design_resi)
show lines, protected_resi
orient design_window
zoom design_window, 6
turn y, -12
turn x, 8
show surface, design_window
set transparency, 0.90, design_window
color mf_surface, design_window
scene design_window, store
png F:/4-15Marsprotein/mars_stack/outputs/paper_bundle_v1/structure_panels/1LBT/design_window.png, width=2400, height=1800, dpi=300, ray=1

save F:/4-15Marsprotein/mars_stack/outputs/paper_bundle_v1/structure_panels/1LBT/figure_session.pse
quit
