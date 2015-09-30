// vi: set expandtab ts=4 sw=4:
#include "ChangeTracker.h"

namespace basegeom {

const std::string ChangeTracker::REASON_ACTIVE_COORD_SET("active_coord_set changed");
const std::string ChangeTracker::REASON_ALT_LOC("alt_loc changed");
const std::string ChangeTracker::REASON_ANISO_U("aniso_u changed");
const std::string ChangeTracker::REASON_BALL_SCALE("ball_scale changed");
const std::string ChangeTracker::REASON_BFACTOR("bfactor changed");
const std::string ChangeTracker::REASON_COLOR("color changed");
const std::string ChangeTracker::REASON_COORD("coord changed");
const std::string ChangeTracker::REASON_DISPLAY("display changed");
const std::string ChangeTracker::REASON_DRAW_MODE("draw_mode changed");
const std::string ChangeTracker::REASON_HALFBOND("halfbond changed");
const std::string ChangeTracker::REASON_HIDE("hide changed");
const std::string ChangeTracker::REASON_IDATM_TYPE("idatm_type changed");
const std::string ChangeTracker::REASON_IS_BACKBONE("is_backbone changed");
const std::string ChangeTracker::REASON_IS_HELIX("is_helix changed");
const std::string ChangeTracker::REASON_IS_HET("is_het changed");
const std::string ChangeTracker::REASON_IS_SHEET("is_sheet changed");
const std::string ChangeTracker::REASON_OCCUPANCY("occupancy changed");
const std::string ChangeTracker::REASON_RADIUS("radius changed");
const std::string ChangeTracker::REASON_RESIDUES("residues changed");
const std::string ChangeTracker::REASON_RIBBON_ADJUST("ribbon_adjust changed");
const std::string ChangeTracker::REASON_RIBBON_COLOR("ribbon_color changed");
const std::string ChangeTracker::REASON_RIBBON_DISPLAY("ribbon_display changed");
const std::string ChangeTracker::REASON_RIBBON_STYLE("ribbon_style changed");
const std::string ChangeTracker::REASON_SELECTED("selected changed");
const std::string ChangeTracker::REASON_SEQUENCE("sequence changed");
const std::string ChangeTracker::REASON_SERIAL_NUMBER("serial_number changed");
const std::string ChangeTracker::REASON_SS_ID("ss_id changed");

DiscardingChangeTracker  dct;

DiscardingChangeTracker*
DiscardingChangeTracker::discarding_change_tracker() { return &dct; }

}  // namespace basegeom
