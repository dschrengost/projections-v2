# Optimizer Ideas & Roadmap

Ideas and future enhancements for the QuickBuild optimizer.

---

## Planned

### Showdown Support
- Single-game showdown slates with Captain/FLEX format
- Different position constraints and salary cap

### Ownership Integration
- Display projected ownership in player pool
- Ownership penalty slider to de-weight chalk
- Leverage targets for contrarian builds

---

## Ideas

### Build Profiles
- Save named presets (e.g., "Cash", "GPP Contrarian", "Max Ceiling")
- Quick-apply profiles from dropdown

### Stacking Rules
- Team stacks (3+ players from same team)
- Game stacks (players from both sides of a game)
- Bring-back rules (pair opposing players)

### Late Swap Mode
- Lock players already in progress
- Rebuild remaining roster spots
- Smart swap suggestions

### Export Enhancements
- Multi-entry support (split builds across entries)
- Diversified upload sets
- Entry group assignments

### Player Exposure
- Set min/max exposure per player across builds
- Visualize exposure distribution

### Correlation Analysis
- Show teammate correlations in pool
- Highlight high-correlation stacks

### Historical Backtesting
- Run optimizer on past slates
- Compare to actual winning lineups
- Track optimizer edge over time

---

## Technical Debt

- [ ] Code-split optimizer page for smaller bundle
- [x] ***Add proper position assignment to CSV export*** These need to at very least output compliant lineups that can be cut/pasted into a draftkings entry csv
- [ ] Add ownership penalty to QuickBuild engine
- [ ] Systemd service for API (currently manual)

---

## Known Issues

### Position Slot Feasibility (Fixed 2024-12-10, Monitor)

**Problem:** The counts-based CP-SAT model (`build_cpsat_counts`) originally used weak position constraints that allowed lineups which couldn't be legally assigned to DK roster slots. Example: a lineup with only one C/PF-eligible player who must fill both C and PF base slots.

**Fix Applied:** Added `>=2` constraints for each base position (PG, SG, SF, PF) to ensure enough players for both base slots and flex slots (G, F).

**Location:** `cpsat_solver.py` lines 605-612

**Risk:** May still produce edge cases on slates with very limited position diversity. If blank slots appear in CSV export, this constraint logic may need further strengthening.

---

## Completed

- [x] Basic lineup generation
- [x] Lock/ban players
- [x] Min/max salary constraints
- [x] Team limits
- [x] Lineup filtering and sorting
- [x] Build persistence (save/load)
- [x] Disk-based slate discovery for past dates


*Daniel's Ideas 12-9-2025*
                                ***Priorities***

***High***
join multiple builds -- user can check multiple builds from the saved builds bar and join them into a single build

display ownership % as a lineup metric in each lineup's card

display / sort / filter by ceiling, projection, ownership % (for entire lineup)


***Medium***
Persist current page state when clicking through tabs (projections and back to optimizer, etc)

remove workers selection from optimizer page, keep in config .yaml (maybe 20-22 threads on 24 thread machine?)

Add ownership min/max to build settings

***Low***





