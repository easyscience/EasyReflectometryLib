# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from easyreflectometry.calculators import CalculatorFactory


def main():
    """Main function."""
    factory = CalculatorFactory()
    print(f'Available calculators: {factory.available_interfaces}')


if __name__ == '__main__':
    main()
