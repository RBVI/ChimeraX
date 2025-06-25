# vim: set expandtab ts=4 sw=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This particular file is part of the ChimeraX library. You can also
# redistribute and/or modify it under the terms of the GNU Lesser General
# Public License version 2.1 as published by the Free Software Foundation.
# For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
# LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
# VERSION 2.1
#
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

# Make a scatter plot for residues using two mutation scores.
def mutation_scores_scatter_plot(session, x_score_name, y_score_name, mutation_set = None,
                                 color_synonymous = True, bounds = True, correlation = False, replace = True):
    
    plot = _find_mutation_scatter_plot(session, mutation_set) if replace else None
    new_plot = (plot is None)
    if new_plot:
        plot = MutationScatterPlot(session)

    from chimerax.core.errors import UserError
    try:
        plot.set_plot_data(x_score_name, y_score_name, mutation_set)
    except UserError:
        # Mutation set or named scores do not exist.
        if new_plot:
            plot.delete()
        raise

    if plot.is_mutation_plot:
        if color_synonymous:
            plot._color_synonymous()
        if bounds:
            plot._show_synonymous_bounds()
                
    if correlation:
        plot.show_least_squares_fit()
        
    message = f'Plotted {len(plot.nodes)} mutations with {x_score_name} on x-axis and {y_score_name} on y-axis'
    if correlation:
        message += f', least squares fit slope {"%.3g" % plot.slope}, intercept {"%.3g" % plot.intercept}, R squared {"%.3g" % plot.r_squared}'
    session.logger.info(message)

from chimerax.interfaces.graph import Graph
class MutationScatterPlot(Graph):

    def __init__(self, session):
        self.mutation_set_name = ''
        self._correlation_shown = False
        self._bounds_artists = []
        self._drag_colors_structure = True
        nodes = edges = []
        Graph.__init__(self, session, nodes, edges,
                       tool_name = 'Mutation scores plot', title = 'Mutation scores plot',
                       hide_ticks = False, drag_select_callback = self._rectangle_selected)

        parent = self.tool_window.ui_area
        layout = parent.layout()

        # Add x-axis, y-axis and mutation set menus.
        from chimerax.ui.widgets import EntriesRow
        menus = EntriesRow(parent,
                           ' X axis', ('score1', 'score2'),
                           'Y axis', ('score1', 'score2'),
                           'Mutations', ('set1', 'set2'))
        self._x_axis_menu, self._y_axis_menu, self._mutation_set_menu = menus.values
        for m in menus.values:
            menu = m.widget.menu()
            menu.aboutToShow.connect(lambda menu=menu: self._menu_about_to_show(menu))
            menu.triggered.connect(self._menu_selection_changed)
        layout.addWidget(menus.frame)

        # Add status line
        from Qt.QtWidgets import QLabel, QSizePolicy
        self._status_line = sl = QLabel(parent)
        sl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        from Qt.QtGui import QFontDatabase
        font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        font.setPointSize(14)
        sl.setFont(font)	# Fixed space font so text maintains alignment
        layout.addWidget(sl)

    def _menu_about_to_show(self, menu):
        menu.clear()
        if menu is self._mutation_set_menu.widget.menu():
            from .ms_data import mutation_scores_list
            for ms_name in mutation_scores_list(self.session):
                menu.addAction(ms_name)
        else:
            from .ms_data import mutation_scores
            ms_name = self._mutation_set_menu.value
            mset = mutation_scores(self.session, ms_name)
            for name in mset.score_names():
                menu.addAction(name)

    def _menu_selection_changed(self, action):
        mutation_set_name = self._mutation_set_menu.value
        x_score_name = self._x_axis_menu.value
        y_score_name = self._y_axis_menu.value

        if mutation_set_name != self.mutation_set_name:
            # If mutation set is changed, make sure x and y-axis score names are valid.
            from .ms_data import mutation_scores
            mset = mutation_scores(self.session, mutation_set_name)
            score_names = mset.score_names()
            if x_score_name not in score_names:
                x_score_name = self._x_axis_menu.value = score_names[0]
            if y_score_name not in score_names:
                y_score_name = self._y_axis_menu.value = score_names[1] if len(score_names) > 1 else score_names[0]
            
        bounds_shown = self._bounds_shown
        preserve_colors = (self.mutation_set_name == mutation_set_name)

        self.set_plot_data(x_score_name, y_score_name, mutation_set_name,
                           color_synonymous = not preserve_colors, preserve_colors = preserve_colors)

        if bounds_shown:
            self._show_synonymous_bounds()


    def set_plot_data(self, x_score_name, y_score_name, mutation_set = None,
                      color_synonymous = False, preserve_colors = False):
        from .ms_data import mutation_scores
        mset = mutation_scores(self.session, mutation_set)
        x_scores = mset.score_values(x_score_name)
        y_scores = mset.score_values(y_score_name)

        self._x_axis_menu.value = x_score_name
        self._y_axis_menu.value = y_score_name
        self._mutation_set_menu.value = self.mutation_set_name = mset.name
        
        points = []
        point_names = []
        if x_scores.per_residue and y_scores.per_residue:
            for res_num in x_scores.residue_numbers():
                x_value = x_scores.residue_value(res_num)
                y_value = y_scores.residue_value(res_num)
                if x_value is not None and y_value is not None:
                    points.append((x_value, y_value))
                    point_names.append(f'{res_num}')
            is_mutation_plot = False	# Residues plotted instead of mutations
        elif x_scores.per_residue or y_scores.per_residue:
            rscores, mscores = (x_scores,y_scores) if x_scores.per_residue else (y_scores,x_scores)
            r_values = {(res_num,from_aa):r_value for res_num, from_aa, to_aa, r_value in rscores.all_values()}
            for res_num, from_aa, to_aa, m_value in mscores.all_values():
                r_value = r_values.get((res_num, from_aa))
                if r_value is not None:
                    points.append((r_value, m_value) if x_scores.per_residue else (m_value, r_value))
                    point_names.append(f'{from_aa}{res_num}{to_aa}')
            is_mutation_plot = True
        else:
            y_values = {(res_num,from_aa,to_aa):y_value for res_num, from_aa, to_aa, y_value in y_scores.all_values()}
            for res_num, from_aa, to_aa, x_value in x_scores.all_values():
                y_value = y_values.get((res_num, from_aa, to_aa))
                if y_value is not None:
                    points.append((x_value, y_value))
                    point_names.append(f'{from_aa}{res_num}{to_aa}')
            is_mutation_plot = True

        from numpy import array, float32
        xy = array(points, float32)

        title = f'File {mset.name}'
        label_nodes, node_area = (False, 20) if is_mutation_plot else (True, 200)

        if preserve_colors:
            from chimerax.core.colors import rgba_to_rgba8
            default_color = rgba_to_rgba8((.8,.8,.8))
            node_colors = {node.description:rgba_to_rgba8(node.color) for node in self.nodes}
            colors = [node_colors.get(name, default_color) for name in point_names]
            node_index = {node.description:i for i,node in enumerate(self.nodes)}
            index = [node_index.get(name, 0) for name in point_names]
            from numpy import argsort
            stack_order = argsort(index)
        else:
            colors = stack_order = None

        self._set_nodes(xy, point_names=point_names, colors=colors, stack_order=stack_order,
                        title=title, x_label=x_score_name, y_label=y_score_name,
                        node_area = node_area, label_nodes = label_nodes, is_mutation_plot = is_mutation_plot)

        if color_synonymous:
            self._color_synonymous()

    def _set_nodes(self, xy, point_names = None, colors = None, stack_order = None,
                   title = '', x_label = '', y_label = '',
                   node_font_size = 5, node_area = 200, label_nodes = True, is_mutation_plot = True):
        self.is_mutation_plot = is_mutation_plot
        self.font_size = node_font_size	# Override graph default value of 12 points
        self.nodes = self._make_nodes(xy, point_names=point_names, colors=colors,
                                      node_area=node_area, label_nodes=label_nodes)
        if stack_order is not None:
            self.nodes = [self.nodes[i] for i in stack_order]  # Last drawn nodes are on top
        self.graph = self._make_graph()
        a = self.axes
        a.clear()
        self.draw_graph()
        a.set_title(title)
        a.set_xlabel(x_label)
        a.set_ylabel(y_label)
        self._show_synonymous_bounds(False)
        self._correlation_shown = False
        self.canvas.draw()

    def show_least_squares_fit(self, xy = None):
        if xy is None:
            x = [node.position[0] for node in self.nodes]
            y = [node.position[1] for node in self.nodes]
        else:
            x, y = xy[:,0], xy[:,1]
        degree = 1
        from numpy import polyfit
        p, ss_r = polyfit(x, y, degree, full=True)[:2]
        fx = (min(x), max(x))
        fy = tuple(p[0]*x + p[1] for x in fx)
        self.axes.plot(fx, fy)
        self.slope, self.intercept, self.r_squared = p[0], p[1], self._r_squared(p, x, y, ss_r)
        self._correlation_shown = correlation

    def _r_squared(self, p, x, y, ss_r):
        r_ys = x*p[0] + p[1]
        from numpy import sum, mean
        ss_tot = sum((y - mean(r_ys)) ** 2)
        r_squared = 1 - (ss_r / ss_tot)
        return r_squared

    def tight_layout(self):
        # Don't hide axes and reduce padding
        pass

    def equal_aspect(self):
        # Don't require both plot axes to have the same scale
        pass

    def _make_nodes(self, xy, point_names = None, colors = None, node_area = 200, label_nodes = True):
        from chimerax.interfaces.graph import Node
        nodes = []
        for i, (x,y) in enumerate(xy):
            n = Node()
            if point_names:
                n.description = point_names[i]
                if label_nodes:
                    n.name = point_names[i]
            n.position = (x, y, 0)
            n.size = node_area
            if colors is not None:
                n.color = tuple(r/255 for r in colors[i])
            nodes.append(n)
        return nodes

    def layout_projection(self):
        from chimerax.geometry import identity
        return identity()

    def mouse_click(self, node, event):
        '''Ctrl click handler.'''
        if node is None:
            self._run_command('select clear')
            self._color_and_raise_nodes([], color = (0,1,0,1), tag = 'sel')
            return
        res = self._node_residues(node)
        if len(res) > 0:
            self._select_residues(res)
            self._color_and_raise_nodes([node], color = (0,1,0,1), tag = 'sel')

    def _node_residues(self, node):
        nres = self._nodes_residues([node])
        from chimerax.atomic import Residues
        res = Residues([r for n,r in nres])
        return res

    def _nodes_residues(self, nodes):
        res_nums = []
        num_to_nodes = {}
        for node in nodes:
            mut_name = node.description
            res_num = int(mut_name[1:-1] if self.is_mutation_plot else mut_name)
            res_nums.append(res_num)
            if res_num in num_to_nodes:
                num_to_nodes[res_num].append(node)
            else:
                num_to_nodes[res_num] = [node]

        from .ms_data import mutation_scores
        mset = mutation_scores(self.session, self.mutation_set_name, raise_error = False)
        if mset is None:
            nres = []
        else:
            mset.associate_chains(self.session)
            res, rnums = mset.associated_residues(res_nums)
            nres = []
            for r, rnum in zip(res, rnums):
                for node in num_to_nodes[rnum]:
                    nres.append((node, r))
        return nres
        
    def mouse_hover(self, event):
        a = self.axes
        xlabel, ylabel = a.get_xlabel(), a.get_ylabel()
        item = self.clicked_item(event)
        if item is not None and hasattr(item, 'description') and hasattr(item, 'position'):
            x,y = item.position[:2]
            descrip = item.description
        else:
            x,y = event.xdata, event.ydata	# Can be None
            descrip = ''
        xval = f'{xlabel} {"%6.2f" % x}' if x is not None else ''
        yval = f'{ylabel} {"%6.2f" % y}' if y is not None else ''
        msg =  f'   {xval}    {yval}    {descrip}'
        self._status_line.setText(msg)

    @property
    def mutation_set(self):
        from .ms_data import mutation_scores
        mset = mutation_scores(self.session, self.mutation_set_name)
        return mset

    def _rectangle_selected(self, event1, event2):
        x1, y1, x2, y2 = event1.xdata, event1.ydata, event2.xdata, event2.ydata
        xmin, xmax = min(x1,x2), max(x1,x2)
        ymin, ymax = min(y1,y2), max(y1,y2)
        rnodes = []
        for node in self.nodes:
            x,y,z = node.position
            if x >= xmin and x <= xmax and y >= ymin and y <= ymax:
                rnodes.append(node)

        cmds = []
        if len(rnodes) > 0:
            nres = self._nodes_residues(rnodes)
            if len(nres) > 0:
                from chimerax.atomic import Residues, concise_residue_spec
                res = Residues(tuple(set(r for node,r in nres)))	# Unique residues
                rspec = concise_residue_spec(self.session, res)
                cmds = [f'select {rspec}']
                if self._drag_colors_structure and len(res) > 0:
                    cspec = res[0].chain.string(style = 'command')
                    cmds.append(f'color {cspec} lightgray ; color {rspec} lime')
        else:
            cmds = ['select clear']
            if self._drag_colors_structure:
                mset = self.mutation_set
                if mset:
                    chains = mset.associated_chains()
                    if chains:
                        from chimerax.atomic import concise_chain_spec
                        cspec = concise_chain_spec(chains)
                        cmds.append(f'color {cspec} lightgray')
        for cmd in cmds:
            self._run_command(cmd)

        # The matplotlig RectangleSelector with useblit restores old matplotlib artists after the selection
        # callback and our coloring code replaces the node drawing artist, so matplotlib brings it back
        # to life.  So we need to delay the coloring until later.
        #self._color_and_raise_nodes(rnodes, color = (0,1,0,1), tag = 'sel')
        t = self.session.ui.timer(0, self._color_and_raise_nodes, rnodes, color = (0,1,0,1), tag = 'sel')
        self._keep_timer_alive = t

    def fill_context_menu(self, menu, item):
        if item is not None:
            r = self._node_residues(item)
            name = item.description
            rname = name[:-1]
            if self.is_mutation_plot:
                self.add_menu_entry(menu, f'Mutation {name}', lambda: None)
                self.add_menu_entry(menu, f'Color mutations for {rname}',
                                    lambda self=self, rname=rname: self._color_residue_mutations(rname))
                if len(r) > 0:
                    self.add_menu_entry(menu, f'Color mutations near residue {rname}',
                                        lambda self=self, r=r: self._color_near(r))
            elif len(r) > 0:
                self.add_menu_entry(menu, f'Color residues near {name}',
                                    lambda self=self, r=r: self._color_near(r))
        if self.is_mutation_plot:
            self.add_menu_entry(menu, 'Color mutations for selected residues', self._color_selected)
            self.add_menu_entry(menu, 'Color synonymous mutations blue', self._color_synonymous)
            show = (len(self._bounds_artists) == 0)
            show_or_hide = 'Show' if show else 'Hide'
            self.add_menu_entry(menu, f'{show_or_hide} synonymous bounds', lambda show=show: self._show_synonymous_bounds(show))
        else:
            self.add_menu_entry(menu, 'Color selected residues on plot', self._color_selected)
        a = self.add_menu_entry(menu, f'Ctrl-drag colors structure', self._toggle_drag_colors_structure)
        a.setCheckable(True)
        a.setChecked(self._drag_colors_structure)
        self.add_menu_entry(menu, f'Clear plot colors', self._clear_colors)

        if item is not None:
            self.add_menu_separator(menu)
            if len(r) == 0:
                self.add_menu_entry(menu, f'{rname} residue not in structure', lambda: None)
            else:
                self.add_menu_entry(menu, f'Structure residue {rname}', lambda: None)
                self.add_menu_entry(menu, f'Select',
                                    lambda self=self, r=r: self._select_residues(r))
                self.add_menu_entry(menu, 'Color green',
                                    lambda self=self, r=r, c=(0,1,0,1): self._color_residues(r,c))
                self.add_menu_entry(menu, 'Color to match plot',
                                    lambda self=self, r=r, c=item.color: self._color_residues(r,c))
                self.add_menu_entry(menu, 'Show side chain',
                                    lambda self=self, r=r: self._show_atoms(r))
                self.add_menu_entry(menu, 'Zoom to residue',
                                    lambda self=self, r=r: self._zoom_to_residues(r))
                if self.is_mutation_plot:
                    a = self.axes
                    xlabel, ylabel = a.get_xlabel(), a.get_ylabel()
                    self.add_menu_entry(menu, f'Label with {xlabel} scores',
                                        lambda self=self, r=r, xlabel=xlabel: self._label(r, xlabel))
                    self.add_menu_entry(menu, f'Label with {ylabel} scores',
                                        lambda self=self, r=r, ylabel=ylabel: self._label(r, ylabel))
                    

        self.add_menu_separator(menu)
        self.add_menu_entry(menu, 'New plot', self._copy_plot)
        self.add_menu_entry(menu, 'Show histogram', self._show_histogram)
        self.add_menu_entry(menu, 'Save plot as...', self.save_plot_as)

    def _select_residues(self, r):
        self._run_residues_command(r, 'select %s')
    def _color_residues(self, r, color):
        from chimerax.core.colors import hex_color, rgba_to_rgba8
        cname = hex_color(rgba_to_rgba8(color))
        self._run_residues_command(r, f'color %s {cname}')
        self._run_residues_command(r, f'color %s byhet')
    def _show_atoms(self, r):
        self._run_residues_command(r, 'show %s atoms')
    def _zoom_to_residues(self, r):
        self._run_residues_command(r, 'view %s')
    def _label(self, r, score_name):
        self._run_residues_command(r, f'mutationscores label %s {score_name}')
    def _color_residue_mutations(self, rname, color = (0,1,0,1)):
        rnodes = [node for node in self.nodes if node.description[:-1] == rname]
        self._color_and_raise_nodes(rnodes, color, tag = 'res')
    def _color_near(self, residues, distance = 3.5):
        cres = set()
        for r in residues:
            cres.update(_find_close_residues(r, r.chain.existing_residues, distance))
        for r in residues:
            cres.discard(r)
        nres = self._nodes_residues(self.nodes)
        nnodes = [node for node,r in nres if r in cres]
        color_names = ['red', 'orange', 'yellow', 'violet', 'magenta', 'salmon', 'seagreen', 'skyblue', 'gold', 'coral']
        from chimerax.core.colors import BuiltinColors
        colors = [BuiltinColors[name].rgba for name in color_names]
        n = len(colors)
        nnames = [node.description[:-1] for node in nnodes]
        nci = {name:i for i,name in enumerate(set(nnames))}
        ncolor = {node: colors[nci[name]%n] for name,node in zip(nnames,nnodes)}
        for node in nnodes:
            node.color = ncolor[node]
            node.color_source = None
        self._color_and_raise_nodes(nnodes, color = None, tag = 'near')
    def _color_synonymous(self, color = (0,0,1,1)):
        syn = [node for node in self.nodes if (node.description[0] == node.description[-1])]
        self._color_and_raise_nodes(syn, color)
    def _color_and_raise_nodes(self, nodes, color, tag = None, uncolor = (.8,.8,.8,1)):
        if tag is not None:
            for node in self.nodes:
                if getattr(node, 'color_source', None) == tag:
                    node.color = uncolor
                    node.color_source = None
        for node in nodes:
            if color is not None:
                node.color = color
            node.color_source = tag
        # Put the colored nodes first so they are drawn on top
        nodeset = set(nodes)
        self.nodes.sort(key = lambda n: 1 if n in nodeset else 0)
        self.graph = self._make_graph()  # Remake graph to get new node order
        self.draw_graph()
        self.canvas.draw()
    def _color_selected(self, color = (0,1,1,1)):
        nres = self._nodes_residues(self.nodes)
        sel = [node for node,r in nres if r and r.selected]
        self._color_and_raise_nodes(sel, color, tag = 'sel')
    def _clear_colors(self, clear_color = (.8,.8,.8,1)):
        self._color_and_raise_nodes(self.nodes, clear_color)
    def _show_synonymous_bounds(self, show = True):
        if (show and self._bounds_shown) or (not show and not self._bounds_shown):
            return  # Already shown or hidden
        if show:
            x0, x1, y0, y1 = self._synonymous_bounds()
            if x0 is None:
                return  # No synonymous mutations
            xmin, xmax, ymin, ymax = self._node_bounds()
            segment_xy = [((x0,x0), (ymin,ymax)), ((x1,x1), (ymin,ymax)), ((xmin,xmax),(y0,y0)), ((xmin,xmax),(y1,y1))]
            from matplotlib.lines import Line2D
            lines = [Line2D(xdata, ydata, color = 'black', linestyle = 'dotted', zorder = 10) for xdata, ydata in segment_xy]
            self._bounds_artists = [self.axes.add_artist(line) for line in lines]
        else:
            for ba in self._bounds_artists:
                if ba.axes is not None:
                    ba.remove()
            self._bounds_artists = []
        self.canvas.draw()
    @property
    def _bounds_shown(self):
        return len(self._bounds_artists) > 0
    def _synonymous_bounds(self, standard_deviations = 2.0):
        syn = [node for node in self.nodes if (node.description[0] == node.description[-1])]
        if len(syn) == 0:
            xmin = xmax = ymin = ymax = None
        else:
            x,y = [node.position[0] for node in syn], [node.position[1] for node in syn]
            from numpy import mean, std
            mx, my, dx, dy = mean(x), mean(y), std(x), std(y)
            sd = standard_deviations
            xmin, xmax, ymin, ymax = mx - sd*dx, mx + sd*dx, my - sd*dy, my + sd*dy
        return xmin, xmax, ymin, ymax
    def _node_bounds(self):
        if len(self.nodes) == 0:
            xmin = xmax = ymin = ymax = None
        else:
            x,y = [node.position[0] for node in self.nodes], [node.position[1] for node in self.nodes]
            xmin, xmax, ymin, ymax = min(x), max(x), min(y), max(y)
        return xmin, xmax, ymin, ymax
    def _copy_plot(self):
        copy = MutationScatterPlot(self.session)
        copy.nodes = self.nodes  # Used for preserving colors
        mutation_set_name = self._mutation_set_menu.value
        x_score_name = self._x_axis_menu.value
        y_score_name = self._y_axis_menu.value
        copy.set_plot_data(x_score_name, y_score_name, mutation_set = mutation_set_name, preserve_colors = True)
        if self._bounds_shown:
            copy._show_synonymous_bounds()
        return copy
    def _show_histogram(self):
        score_name = self._x_axis_menu.value
        mset_name = self._mutation_set_menu.value
        self._run_command(f'mutationscores histogram {score_name} mutationSet {mset_name}')
    def _toggle_drag_colors_structure(self):
        self._drag_colors_structure = not self._drag_colors_structure
    def _run_residues_command(self, res, command):
        from chimerax.atomic import concise_residue_spec
        rspec = concise_residue_spec(self.session, res)
        self._run_command(command % rspec)
    def _run_command(self, command):
        from chimerax.core.commands import run
        run(self.session, command)
    
    # ---------------------------------------------------------------------------
    # Session save and restore.
    #
    SESSION_SAVE = True
    def take_snapshot(self, session, flags):
        xy, point_names, colors, node_area, label_nodes = [], [], [], 200, False
        from chimerax.core.colors import rgba_to_rgba8
        for node in self.nodes:
            xy.append(node.position[:2])
            node_area = node.size
            if hasattr(node, 'description'):
                point_names.append(node.description)
            if node.name:
                label_nodes = True
            if hasattr(node, 'color'):
                colors.append(rgba_to_rgba8(node.color))
        axes = self.axes
        data = {'mutation_set_name': self.mutation_set_name,
                'xy': xy,
                'point_names': (None if len(point_names) == 0 else point_names),
                'colors': (None if len(colors) == 0 else colors),
                'correlation': self._correlation_shown,
                'title': axes.get_title(),
                'x_label': axes.get_xlabel(),
                'y_label': axes.get_ylabel(),
                'font_size': self.font_size,
                'node_area': node_area,
                'label_nodes': label_nodes,
                'is_mutation_plot': self.is_mutation_plot,
                'bounds_shown': self._bounds_shown,
                'version': '1'}
        return data

    @classmethod
    def restore_snapshot(cls, session, data):
        sp = cls(session)
        sp.mutation_set_name = sp._mutation_set_menu.value = data['mutation_set_name']
        sp._x_axis_menu.value = data.get('x_label')
        sp._y_axis_menu.value = data.get('y_label')
        sp._set_nodes(data['xy'], point_names = data['point_names'], colors = data['colors'],
                      title = data['title'], x_label = data['x_label'], y_label = data['y_label'],
                      node_font_size = data['font_size'], node_area = data['node_area'], label_nodes = data['label_nodes'],
                      is_mutation_plot = data['is_mutation_plot'])
        sp._show_synonymous_bounds(data.get('bounds_shown'))
        if data['correlation']:
            sp.show_least_squares_fit()
        return sp

def _find_close_residues(residue, residues, distance):
    rxyz = residue.atoms.coords
    aatoms = residues.atoms
    axyz = aatoms.coords
    from chimerax.geometry import find_close_points
    ri, ai = find_close_points(rxyz, axyz, distance)
    close_res = aatoms[ai].residues.unique()
    return close_res

def _find_mutation_scatter_plot(session, mutation_set_name = None):
    plots = [tool for tool in session.tools.list()
             if isinstance(tool, MutationScatterPlot) and (mutation_set_name is None or tool.mutation_set_name == mutation_set_name)]
    return plots[-1] if plots else None

def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, StringArg, BoolArg
    desc = CmdDesc(
        required = [('x_score_name', StringArg),
                    ('y_score_name', StringArg)],
        keyword = [('mutation_set', StringArg),
                   ('color_synonymous', BoolArg),
                   ('bounds', BoolArg),
                   ('correlation', BoolArg),
                   ('replace', BoolArg),
                   ],
        synopsis = 'Show scatter plot of residues using two mutation scores'
    )
    register('mutationscores scatterplot', desc, mutation_scores_scatter_plot, logger=logger)
