# SPDX-FileCopyrightText: 2024 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from html import escape
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version
from urllib.parse import quote

import matplotlib.pyplot as plt
import numpy as np
from easyscience import global_object
from xhtml2pdf import pisa

from easyreflectometry import Project

from .html_templates import HTML_DATA_COLLECTION_TEMPLATE
from .html_templates import HTML_FIGURES_TEMPLATE
from .html_templates import HTML_PARAMETER_HEADER_TEMPLATE
from .html_templates import HTML_PARAMETER_TEMPLATE
from .html_templates import HTML_PROJECT_INFORMATION_TEMPLATE
from .html_templates import HTML_REFINEMENT_TEMPLATE
from .html_templates import HTML_TEMPLATE

_NAME_MAX_LEN = 20
# Custom href scheme used to pass the full name to QML via TextEdit.hoveredLink.
_TOOLTIP_SCHEME = 'nametooltip'

# URLs for known calculation engines and minimizer packages.
_ENGINE_URLS: dict[str, str] = {
    'refnx': 'https://refnx.readthedocs.io',
    'refl1d': 'https://refl1d.readthedocs.io',
    'bornagain': 'https://www.bornagainproject.org',
    'lm': 'https://lmfit.github.io/lmfit-py/',
    'bumps': 'https://bumps.readthedocs.io',
    'dfo': 'https://github.com/fitbenchmarking/dfo-ls',
}


def _engine_link(name: str, package: str | None = None) -> str:
    """Return an HTML hyperlink for an engine, including its version.

    Falls back to plain text when no URL is known for the engine.
    """
    url = _ENGINE_URLS.get(name) or _ENGINE_URLS.get(package or '')
    display = escape(name)
    if package:
        try:
            ver = version(package)
        except PackageNotFoundError:
            ver = None
        if ver:
            display = f'{display} (v{ver})'

    if url:
        return f'<a href="{url}">{display}</a>'
    return display


def _format_value(value: float, sig_figs: int) -> str:
    """Format a numeric value for summary display.

    *sig_figs* significant figures; fall back to 1-decimal exponential when
    the formatted string is too long.  Zero is shown as '0.0'.
    """
    if value == 0.0:
        return '0.0'
    s = f'{value:.{sig_figs}g}'
    if len(s) <= sig_figs + 4:
        return s
    return f'{value:.1e}'


def _truncate_name(name: str, max_len: int = _NAME_MAX_LEN) -> str:
    """Return an HTML snippet with a truncated name and tooltip for the full text.

    Browsers use the ``title`` attribute; QML reads the ``href`` via
    ``TextEdit.hoveredLink`` and shows a native ToolTip.
    """
    safe = escape(name)
    if len(name) <= max_len:
        return safe
    short = escape(name[:max_len].rstrip())
    encoded = quote(name, safe='')
    return f'<a style="color:inherit;text-decoration:none;" href="{_TOOLTIP_SCHEME}:{encoded}" title="{safe}">{short}…</a>'


class Summary:
    def __init__(self, project: Project):
        """Init function."""
        self._project = project

    def compile_html_summary(self, figures: bool = False) -> str:
        """Compile html summary."""
        html = HTML_TEMPLATE

        html = html.replace('project_information_section', self._project_information_section())

        html = html.replace('sample_section', self._sample_section())

        experiments_section = self._experiments_section()
        if experiments_section == '':  # no experiments
            experiments_section = '<td>No experiments</td>'
        html = html.replace('experiments_section', experiments_section)

        html = html.replace('refinement_section', self._refinement_section())

        if figures:
            html = html.replace('figures_section', self._figures_section())
        else:
            html = html.replace('figures_section', '')

        return html

    def save_html_summary(self, filename: str) -> None:
        """Save html summary."""
        html = self.compile_html_summary(figures=True)
        with open(filename, 'w') as f:
            f.write(html)

    def save_pdf_summary(self, filename: str) -> None:
        """Save pdf summary."""
        html = self.compile_html_summary(figures=True)

        with open(filename, 'w+b') as result_file:
            pisa_status = pisa.CreatePDF(
                html,
                dest=result_file,
            )

            if pisa_status.err:
                print('An error occured when generating PDF summary!')

    def save_sld_plot(self, filename: str) -> None:
        """Save sld plot."""
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        sld = self._project.sld_data_for_model_at_index(0)
        ax.plot(sld.x, sld.y, color='blue')

        ax.set_xlabel('z (Å)')
        ax.set_ylabel('SLD (Å⁻²)')
        fig.legend(['SLD'])
        fig.savefig(filename, dpi=600)
        plt.close()

    def save_fit_experiment_plot(self, filename: str) -> None:
        """Save fit experiment plot."""
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        legends = []

        model = self._project.model_data_for_model_at_index(0)
        ax.plot(model.x, np.log10(model.y), color='blue')
        legends.append('Model')

        try:
            experiment = self._project.experimental_data_for_model_at_index(0)
            ax.plot(experiment.x, np.log10(experiment.y), color='red')
            legends.append('Experiment')
        except IndexError:
            pass

        ax.set_xlabel('Q (Å⁻¹)')
        ax.set_ylabel('Reflectivity')
        fig.legend(legends)
        fig.savefig(filename, dpi=600)
        plt.close()

    def _project_information_section(self) -> str:
        """Project information section."""
        html_project = HTML_PROJECT_INFORMATION_TEMPLATE

        name = self._project._info['name']
        short_description = self._project._info['short_description']
        html_project = html_project.replace('project_title', f'{name}')
        html_project = html_project.replace('project_description', f'{short_description}')
        html_project = html_project.replace('num_experiments', f'{len(self._project.experiments)}')
        return html_project

    def _sample_section(self) -> str:
        """Sample section."""
        html_parameters = []

        html_parameter = HTML_PARAMETER_HEADER_TEMPLATE
        html_parameter = html_parameter.replace('parameter_name', 'Name')
        html_parameter = html_parameter.replace('parameter_value', 'Value')
        html_parameter = html_parameter.replace('parameter_unit', 'Unit')
        html_parameter = html_parameter.replace('parameter_error', 'Error')
        html_parameters.append(html_parameter)

        # Get parameters directly from the model instead of using project.parameters
        model = self._project._models[self._project.current_model_index]
        parameters = model.get_parameters()

        for parameter in parameters:
            path = global_object.map.find_path(model.unique_name, parameter.unique_name)
            if 0 < len(path):
                name = f'{global_object.map.get_item_by_key(path[-2]).name} {global_object.map.get_item_by_key(path[-1]).name}'
            else:
                name = parameter.name
            value = parameter.value
            unit = parameter.unit
            error = parameter.error

            html_parameter = HTML_PARAMETER_TEMPLATE
            html_parameter = html_parameter.replace('parameter_name', f'{name}')
            html_parameter = html_parameter.replace('parameter_value', _format_value(value, 3))
            html_parameter = html_parameter.replace('parameter_unit', f'{unit}')
            error_str = _format_value(error, 2)
            html_parameter = html_parameter.replace('parameter_error', error_str)
            html_parameters.append(html_parameter)

        html_parameters_str = '\n'.join(html_parameters)

        return html_parameters_str

    def _experiments_section(self) -> str:
        """Experiments section."""
        html_experiments = []

        for idx, experiment in self._project.experiments.items():
            experiment_name = experiment.name
            num_data_points = len(experiment.x)
            resolution_function = experiment.model.resolution_function.as_dict()['smearing']
            if resolution_function == 'PercentageFwhm':
                precentage = experiment.model.resolution_function.as_dict()['constant']
                resolution_function = f'{resolution_function} {precentage}%'
            range_min = min(experiment.y)
            range_max = max(experiment.y)
            range_units = 'Å⁻¹'
            html_experiment = HTML_DATA_COLLECTION_TEMPLATE
            html_experiment = html_experiment.replace('experiment_name', _truncate_name(experiment_name))
            html_experiment = html_experiment.replace('range_min', _format_value(range_min, 2))
            html_experiment = html_experiment.replace('range_max', _format_value(range_max, 2))
            html_experiment = html_experiment.replace('range_units', f'{range_units}')
            html_experiment = html_experiment.replace('num_data_points', f'{num_data_points}')
            html_experiment = html_experiment.replace('resolution_function', f'{resolution_function}')
            html_experiments.append(html_experiment)

        html_experiments_str = '\n'.join(html_experiments)

        return html_experiments_str

    def _refinement_section(self) -> str:
        """Refinement section."""
        html_refinement = HTML_REFINEMENT_TEMPLATE

        # Get parameters directly from the model
        model = self._project._models[self._project.current_model_index]
        parameters = model.get_parameters()

        num_free_params = sum(1 for parameter in parameters if parameter.free)
        num_fixed_params = sum(1 for parameter in parameters if not parameter.free)
        num_params = num_free_params + num_fixed_params
        num_constraints = sum(1 for parameter in parameters if not parameter.independent)

        goodness_of_fit = self._compute_goodness_of_fit()

        html_refinement = html_refinement.replace(
            'calculation_engine',
            _engine_link(self._project._calculator.current_interface_name),
        )
        html_refinement = html_refinement.replace(
            'minimization_engine',
            _engine_link(self._project.minimizer.name, self._project.minimizer.package),
        )
        html_refinement = html_refinement.replace('goodness_of_fit', goodness_of_fit)
        html_refinement = html_refinement.replace('num_total_params', f'{num_params}')
        html_refinement = html_refinement.replace('num_free_params', f'{num_free_params}')
        html_refinement = html_refinement.replace('num_fixed_params', f'{num_fixed_params}')
        html_refinement = html_refinement.replace('num_constriants', f'{num_constraints}')
        return html_refinement

    def _compute_goodness_of_fit(self) -> str:
        """Return reduced chi² as a formatted string, or 'N/A' if no fit has been run."""
        last_fit_results = getattr(self._project, '_last_fit_results', None)
        if not last_fit_results:
            return 'N/A'
        try:
            if len(last_fit_results) == 1:
                gof = float(last_fit_results[0].reduced_chi2)
            else:
                total_chi2 = sum(float(r.chi2) for r in last_fit_results)
                total_points = sum(len(r.x) for r in last_fit_results)
                n_pars = last_fit_results[0].n_pars
                dof = total_points - n_pars
                gof = total_chi2 / dof if dof > 0 else 0.0
            return f'{gof:.4g}'
        except (AttributeError, TypeError, ValueError, ZeroDivisionError):
            return 'N/A'

    def _figures_section(self) -> None:
        """Figures section."""
        html_figures = HTML_FIGURES_TEMPLATE
        path_sld = self._project.path / 'sld_plot.jpg'
        path_fit_experiment = self._project.path / 'fit_experiment_plot.jpg'

        self.save_sld_plot(path_sld)
        self.save_fit_experiment_plot(path_fit_experiment)

        html_figures = html_figures.replace('path_sld_plot', str(path_sld))
        html_figures = html_figures.replace('path_fit_experiment_plot', str(path_fit_experiment))
        return html_figures
