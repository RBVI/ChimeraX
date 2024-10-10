# Make a scatter plot for residues using two phenotype scores from deep mutational scan data.
def dms_scatter_plot(session, chain, x_column_name, y_column_name,
                     correlation = False, subtract_x_fit = None, subtract_y_fit = None,
                     type = 'all_mutations', above = None, below = None, replace = True):
    from .dms_data import dms_data
    data = dms_data(chain)
    if data is None:
        from chimerax.core.errors import UserError
        raise UserError(f'No deep mutation scan data associated with chain {chain}')
    x_scores = data.column_values(x_column_name, subtract_fit = subtract_x_fit)
    y_scores = data.column_values(y_column_name, subtract_fit = subtract_y_fit)
    
    res_nums = []
    points = []
    point_names = []
    if type == 'all_mutations':
        y_values = {(res_num,from_aa,to_aa):y_value for res_num, from_aa, to_aa, y_value in y_scores.all_values()}
        for res_num, from_aa, to_aa, x_value in x_scores.all_values():
            y_value = y_values.get((res_num, from_aa, to_aa))
            if y_value is not None:
                res_nums.append(res_num)
                points.append((x_value, y_value))
                point_names.append(f'{from_aa}{res_num}{to_aa}')
    else:
        for res_num in x_scores.residue_numbers():
            x_value = x_scores.residue_value(res_num, value_type = type, above = above, below = below)
            y_value = y_scores.residue_value(res_num, value_type = type, above = above, below = below)
            if x_value is not None and y_value is not None:
                res_nums.append(res_num)
                points.append((x_value, y_value))
                point_names.append(f'{res_num}')

    resnum_to_res = {r.number:r for r in chain.existing_residues}
    residues = [resnum_to_res.get(res_num) for res_num in res_nums]
    
    from numpy import array, float32
    xy = array(points, float32)
    
    if replace and hasattr(chain, '_last_dms_plot') and chain._last_dms_plot.tool_window.ui_area is not None:
        plot = chain._last_dms_plot
    else:
        chain._last_dms_plot = plot = ResidueScatterPlot(session)
    title = f'File {data.name}'
    label_nodes, node_area = (False, 20) if type == 'all_mutations' else (True, 200)
    plot.set_nodes(xy, residues, point_names=point_names, correlation=correlation,
                   title=title, x_label=x_column_name, y_label=y_column_name,
                   node_area = node_area, label_nodes = label_nodes)
    
    message = f'Plotted {len(points)} mutations in chain {chain} with {x_column_name} on x-axis and {y_column_name} on y-axis'
    if correlation:
        message += f', least squares fit slope {"%.3g" % plot.slope}, intercept {"%.3g" % plot.intercept}, R squared {"%.3g" % plot.r_squared}'
    session.logger.info(message)

from chimerax.interfaces.graph import Graph
class ResidueScatterPlot(Graph):

    def __init__(self, session):
        nodes = edges = []
        Graph.__init__(self, session, nodes, edges, tool_name = 'DeepMutationalScan',
                       title = 'Deep mutational scan scatter plot', hide_ticks = False)
        self._highlight_color = (0,255,0,255)
        self._unhighlight_color = (150,150,150,255)

    def set_nodes(self, xy, residues, point_names = None, colors = None, correlation = False,
                  title = '', x_label = '', y_label = '',
                  node_font_size = 5, node_area = 200, label_nodes = True):
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
            return
        r = node.residue
        if r is not None and not r.deleted:
            r.chain.existing_residues.ribbon_colors = self._unhighlight_color
            r.ribbon_color = self._highlight_color
            r.atoms.colors = self._highlight_color

    def fill_context_menu(self, menu, item):
        if item is not None:
            r = item.residue
            name = item.description
            self.add_menu_entry(menu, f'Mutation {name}', lambda: None)
            self.add_menu_entry(menu, f'Color residue {name[:-1]} on plot',
                                lambda self=self, n=name: self._color_residue_on_plot(name[:-1]))
            self.add_menu_separator(menu)
            if r is None or r.deleted:
                self.add_menu_entry(menu, f'{name} residue not in structure', lambda: None)
            else:
                rname = f'{r.one_letter_code}{r.number}'
                self.add_menu_entry(menu, f'Structure residue {rname}', lambda: None)
                self.add_menu_entry(menu, f'Select', lambda self=self, r=r: self._select_residue(r))
                self.add_menu_entry(menu, f'Color green', lambda self=self, r=r: self._highlight_residue(r))
                self.add_menu_entry(menu, f'Zoom to residue', lambda self=self, r=r: self._zoom_to_residue(r))
        else:
            self.add_menu_entry(menu, 'Save Plot As...', self.save_plot_as)

    def _select_residue(self, r):
        self._run_residue_command(r, 'select %s')
    def _highlight_residue(self, r):
        self._run_residue_command(r, 'color %s lime')
    def _zoom_to_residue(self, r):
        self._run_residue_command(r, 'view %s')
    def _color_residue_on_plot(self, r_name_num):
        highlight_color = [r/255 for r in self._highlight_color]
        for node in self.nodes:
            if node.description[:-1] == r_name_num:
                if not hasattr(node, 'original_color'):
                    node.original_color = node.color
                node.color = highlight_color
            elif hasattr(node, 'original_color'):
                node.color = node.original_color
        # Put the colored nodes first so they are drawn on top
        self.nodes.sort(key = lambda n: 1 if n.description[:-1] == r_name_num else 0)
        self.graph = self._make_graph()  # Remake graph to get new node order
        self.draw_graph()
    def _run_residue_command(self, r, command):
        self._run_command(command % r.string(style = 'command'))
    def _run_command(self, command):
        from chimerax.core.commands import run
        run(self.session, command)

def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, StringArg, EnumOf, BoolArg, FloatArg
    from chimerax.atomic import ChainArg
    from .dms_data import ColumnValues
    desc = CmdDesc(
        required = [('chain', ChainArg)],
        keyword = [('x_column_name', StringArg),
                   ('y_column_name', StringArg),
                   ('correlation', BoolArg),
                   ('subtract_x_fit', StringArg),
                   ('subtract_y_fit', StringArg),
                   ('type', EnumOf(('all_mutations',) + ColumnValues.residue_value_types)),
                   ('above', FloatArg),
                   ('below', FloatArg),
                   ('replace', BoolArg),
                   ],
        required_arguments = ['x_column_name', 'y_column_name'],
        synopsis = 'Show scatter plot of residues using two phenotype deep mutational scan scores'
    )
    register('dms scatterplot', desc, dms_scatter_plot, logger=logger)
