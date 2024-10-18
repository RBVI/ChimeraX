# Make a scatter plot for residues using two mutation scores.
def mutation_scores_scatter_plot(session, x_score_name, y_score_name, mutation_set = None,
                                 correlation = False, replace = True):
    from .ms_data import mutation_scores
    scores = mutation_scores(session, mutation_set)
    x_scores = scores.score_values(x_score_name)
    y_scores = scores.score_values(y_score_name)
    
    res_nums = []
    points = []
    point_names = []
    if x_scores.per_residue and y_scores.per_residue:
        for res_num in x_scores.residue_numbers():
            x_value = x_scores.residue_value(res_num)
            y_value = y_scores.residue_value(res_num)
            if x_value is not None and y_value is not None:
                res_nums.append(res_num)
                points.append((x_value, y_value))
                point_names.append(f'{res_num}')
        is_mutation_plot = False	# Residues plotted instead of mutations
    elif x_scores.per_residue or y_scores.per_residue:
        rscores, mscores = (x_scores,y_scores) if x_scores.per_residue else (y_scores,x_scores)
        r_values = {(res_num,from_aa):r_value for res_num, from_aa, to_aa, r_value in rscores.all_values()}
        for res_num, from_aa, to_aa, m_value in mscores.all_values():
            r_value = r_values.get((res_num, from_aa))
            if r_value is not None:
                res_nums.append(res_num)
                points.append((r_value, m_value) if x_scores.per_residue else (m_value, r_value))
                point_names.append(f'{from_aa}{res_num}{to_aa}')
        is_mutation_plot = True
    else:
        y_values = {(res_num,from_aa,to_aa):y_value for res_num, from_aa, to_aa, y_value in y_scores.all_values()}
        for res_num, from_aa, to_aa, x_value in x_scores.all_values():
            y_value = y_values.get((res_num, from_aa, to_aa))
            if y_value is not None:
                res_nums.append(res_num)
                points.append((x_value, y_value))
                point_names.append(f'{from_aa}{res_num}{to_aa}')
        is_mutation_plot = True

    chain = scores.chain
    if chain is None:
        chain = scores.find_matching_chain(session)
    if chain:
        resnum_to_res = {r.number:r for r in chain.existing_residues}
        residues = [resnum_to_res.get(res_num) for res_num in res_nums]
        in_struct = f' in chain {chain}'
    else:
        residues = [None] * len(res_nums)
        in_struct = ''
    
    from numpy import array, float32
    xy = array(points, float32)
    
    if replace:
        plot = getattr(scores, '_last_mutation_scores_plot', None)
        if plot and plot.tool_window.ui_area is None:
            plot = None
    if plot is None:
        plot = ResidueScatterPlot(session)
    scores._last_mutation_scores_plot = plot

    title = f'File {scores.name}'
    label_nodes, node_area = (False, 20) if is_mutation_plot else (True, 200)
    plot.set_nodes(xy, residues, point_names=point_names, correlation=correlation,
                   title=title, x_label=x_score_name, y_label=y_score_name,
                   node_area = node_area, label_nodes = label_nodes, is_mutation_plot = is_mutation_plot)
    
    message = f'Plotted {len(points)} mutations{in_struct} with {x_score_name} on x-axis and {y_score_name} on y-axis'
    if correlation:
        message += f', least squares fit slope {"%.3g" % plot.slope}, intercept {"%.3g" % plot.intercept}, R squared {"%.3g" % plot.r_squared}'
    session.logger.info(message)

from chimerax.interfaces.graph import Graph
class ResidueScatterPlot(Graph):

    def __init__(self, session):
        nodes = edges = []
        Graph.__init__(self, session, nodes, edges, tool_name = 'DeepMutationalScan',
                       title = 'Deep mutational scan scatter plot', hide_ticks = False,
                       drag_select_callback = self._rectangle_selected)

        # Add status line
        parent = self.tool_window.ui_area
        from Qt.QtWidgets import QLabel, QSizePolicy
        self._status_line = sl = QLabel(parent)
        sl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        from Qt.QtGui import QFontDatabase
        font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        font.setPointSize(14)
        sl.setFont(font)	# Fixed space font so text maintains alignment
        parent.layout().addWidget(sl)

    def set_nodes(self, xy, residues, point_names = None, colors = None, correlation = False,
                  title = '', x_label = '', y_label = '',
                  node_font_size = 5, node_area = 200, label_nodes = True, is_mutation_plot = True):
        self.is_mutation_plot = is_mutation_plot
        self.font_size = node_font_size	# Override graph default value of 12 points
        self.nodes = self._make_nodes(xy, residues, point_names=point_names, colors=colors,
                                      node_area=node_area, label_nodes=label_nodes)
        self.graph = self._make_graph()
        a = self.axes
        a.clear()
        self.draw_graph()
        a.set_title(title)
        a.set_xlabel(x_label)
        a.set_ylabel(y_label)
        if correlation:
            self.show_least_squares_fit(xy)
        self.canvas.draw()

    def show_least_squares_fit(self, xy):
        x, y = xy[:,0], xy[:,1]
        degree = 1
        from numpy import polyfit
        p, ss_r = polyfit(x, y, degree, full=True)[:2]
        fx = (min(x), max(x))
        fy = tuple(p[0]*x + p[1] for x in fx)
        self.axes.plot(fx, fy)
        self.slope, self.intercept, self.r_squared = p[0], p[1], self._r_squared(p, x, y, ss_r)

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

    def _make_nodes(self, xy, residues, point_names = None, colors = None, node_area = 200, label_nodes = True):
        from chimerax.interfaces.graph import Node
        nodes = []
        for i, (res, (x,y)) in enumerate(zip(residues, xy)):
            n = Node()
            if point_names:
                n.description = point_names[i]
                if label_nodes:
                    n.name = point_names[i]
            n.position = (x, y, 0)
            n.size = node_area
            if colors is not None:
                n.color = tuple(r/255 for r in colors[i])
            n.residue = res
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
        r = node.residue
        if r is not None and not r.deleted:
            self._select_residue(r)
            self._color_and_raise_nodes([node], color = (0,1,0,1), tag = 'sel')

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

    def _rectangle_selected(self, event1, event2):
        x1, y1, x2, y2 = event1.xdata, event1.ydata, event2.xdata, event2.ydata
        xmin, xmax = min(x1,x2), max(x1,x2)
        ymin, ymax = min(y1,y2), max(y1,y2)
        rnodes = []
        for node in self.nodes:
            x,y,z = node.position
            if x >= xmin and x <= xmax and y >= ymin and y <= ymax:
                rnodes.append(node)

        if len(rnodes) > 0:
            from chimerax.atomic import Residues, concise_residue_spec
            res = Residues(tuple(set([node.residue for node in rnodes if node.residue is not None])))
            rspec = concise_residue_spec(self.session, res)
            cmd = f'select {rspec}'
        else:
            cmd = 'select clear'
        self._run_command(cmd)

        # The matplotlig RectangleSelector with useblit restores old matplotlib artists after the selection
        # callback and our coloring code replaces the node drawing artist, so matplotlib brings it back
        # to life.  So we need to delay the coloring until later.
        #self._color_and_raise_nodes(rnodes, color = (0,1,0,1), tag = 'sel')
        t = self.session.ui.timer(0, self._color_and_raise_nodes, rnodes, color = (0,1,0,1), tag = 'sel')
        self._keep_timer_alive = t

    def fill_context_menu(self, menu, item):
        if item is not None:
            r = item.residue
            name = item.description
            rname = name[:-1]
            if self.is_mutation_plot:
                self.add_menu_entry(menu, f'Mutation {name}', lambda: None)
                self.add_menu_entry(menu, f'Color mutations for {rname}',
                                    lambda self=self, rname=rname: self._color_residue_mutations(rname))
                if r:
                    self.add_menu_entry(menu, f'Color mutations near residue {rname}',
                                        lambda self=self, r=r: self._color_near(r))
            elif r:
                self.add_menu_entry(menu, f'Color residues near {name}',
                                    lambda self=self, r=r: self._color_near(r))
        if self.is_mutation_plot:
            self.add_menu_entry(menu, f'Color mutations for selected residues', self._color_selected)
            self.add_menu_entry(menu, f'Color synonymous mutations blue', self._color_synonymous)
        else:
            self.add_menu_entry(menu, f'Color selected residues on plot', self._color_selected)
        self.add_menu_entry(menu, f'Clear plot colors', self._clear_colors)

        if item is not None:
            self.add_menu_separator(menu)
            if r is None or r.deleted:
                self.add_menu_entry(menu, f'{rname} residue not in structure', lambda: None)
            else:
                rname = f'{r.one_letter_code}{r.number}'
                self.add_menu_entry(menu, f'Structure residue {rname}', lambda: None)
                self.add_menu_entry(menu, f'Select',
                                    lambda self=self, r=r: self._select_residue(r))
                self.add_menu_entry(menu, f'Color green',
                                    lambda self=self, r=r, c=(0,1,0,1): self._color_residue(r,c))
                self.add_menu_entry(menu, f'Color to match plot',
                                    lambda self=self, r=r, c=item.color: self._color_residue(r,c))
                self.add_menu_entry(menu, f'Show side chain',
                                    lambda self=self, r=r: self._show_atoms(r))
                self.add_menu_entry(menu, f'Zoom to residue',
                                    lambda self=self, r=r: self._zoom_to_residue(r))
                if self.is_mutation_plot:
                    a = self.axes
                    xlabel, ylabel = a.get_xlabel(), a.get_ylabel()
                    self.add_menu_entry(menu, f'Label with {xlabel} scores',
                                        lambda self=self, r=r, xlabel=xlabel: self._label(r, xlabel))
                    self.add_menu_entry(menu, f'Label with {ylabel} scores',
                                        lambda self=self, r=r, ylabel=ylabel: self._label(r, ylabel))
                    

        self.add_menu_separator(menu)                
        self.add_menu_entry(menu, 'Save Plot As...', self.save_plot_as)

    def _select_residue(self, r):
        self._run_residue_command(r, 'select %s')
    def _color_residue(self, r, color):
        from chimerax.core.colors import hex_color, rgba_to_rgba8
        cname = hex_color(rgba_to_rgba8(color))
        self._run_residue_command(r, f'color %s {cname}')
        self._run_residue_command(r, f'color %s byhet')
    def _show_atoms(self, r):
        self._run_residue_command(r, 'show %s atoms')
    def _zoom_to_residue(self, r):
        self._run_residue_command(r, 'view %s')
    def _label(self, r, score_name):
        self._run_residue_command(r, f'mutationscores label %s {score_name}')
    def _color_residue_mutations(self, rname, color = (0,1,0,1)):
        rnodes = [node for node in self.nodes if node.description[:-1] == rname]
        self._color_and_raise_nodes(rnodes, color, tag = 'res')
    def _color_near(self, residue, distance = 3.5):
        cres = set(_find_close_residues(residue, residue.chain.existing_residues, distance))
        nnodes = [node for node in self.nodes if node.residue in cres]
        color_names = ['red', 'orange', 'yellow', 'violet', 'magenta', 'salmon', 'seagreen', 'skyblue', 'gold', 'coral']
        from chimerax.core.colors import BuiltinColors
        colors = [BuiltinColors[name].rgba for name in color_names]
        n = len(colors)
        root_color = (0,1,0,1)
        rcolor = {r:(root_color if r is residue else colors[i%n]) for i,r in enumerate(cres)}
        for node in nnodes:
            node.color = rcolor[node.residue]
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
        sel = [node for node in self.nodes if node.residue and node.residue.selected]
        self._color_and_raise_nodes(sel, color, tag = 'sel')
    def _clear_colors(self, clear_color = (.8,.8,.8,1)):
        self._color_and_raise_nodes(self.nodes, clear_color)
    def _run_residue_command(self, r, command):
        self._run_command(command % r.string(style = 'command'))
    def _run_command(self, command):
        from chimerax.core.commands import run
        run(self.session, command)

def _find_close_residues(residue, residues, distance):
    rxyz = residue.atoms.coords
    aatoms = residues.atoms
    axyz = aatoms.coords
    from chimerax.geometry import find_close_points
    ri, ai = find_close_points(rxyz, axyz, distance)
    close_res = aatoms[ai].residues.unique()
    return close_res

def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, StringArg, BoolArg
    desc = CmdDesc(
        required = [('x_score_name', StringArg),
                    ('y_score_name', StringArg)],
        keyword = [('mutation_set', StringArg),
                   ('correlation', BoolArg),
                   ('replace', BoolArg),
                   ],
        synopsis = 'Show scatter plot of residues using two mutation scores'
    )
    register('mutationscores scatterplot', desc, mutation_scores_scatter_plot, logger=logger)
