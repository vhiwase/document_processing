#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_matrix_rotation
----------------------------------
Tests for `matrix_rotation` module.
"""

import unittest
from click.testing import CliRunner
from document_processing import azure_api_ocr
import pathlib

__all__ = ['TestazureOCR']

FILE_PATH = pathlib.PosixPath(__file__)

BASE = FILE_PATH.parent


class TestazureOCR(unittest.TestCase):


    def setUp(self):        
        self.endpoint = 'https://technomilecognitivieservices.cognitiveservices.azure.com'
        self.subscription_key = '0766c7b636c04a2fad9c90d68f008b8d'
        
        

    document_filepath='/home/vaibhav/Documents/Personal/document_processing/tests/samples/sample2.pdf'
    document_filepath='/home/vaibhav/Documents/Personal/document_processing/tests/samples/sample3.pdf'

    def test_azure_read_api_sample1(self):
        
        document_filepath = (BASE / 'sample1.pdf').absolute().as_posix()
        VERTICAL_COLUMNS=[
            'CLIN', 'ITEM NO', 'SUPPLIES/SERVICES', 'ESTIMATED QUANTITY', 'EST. AMOUNT', 
            'UNIT PRICE', 'DESCRIPTION OF SUPPLIES/SERVICES', "Section", "F Item #", 
            'PWS', 'Deliverable', 'Due Date', 'Addressee and', 'F Item #', 
            'cross-', 'Number/type', 'reference', 'of copies', 'CONTRACT LINE ITEM NUMBER (CLIN)', 'QTY', 'TOTAL']
        HORIZONTAL_COLUMNS=['NET AMT', 'ESTIMATED COST', 'FIXED FEE', 'TOTAL EST COST + FEE']
        HORIZONTAL_KEYWORDS=['ACRN', 'CIN']
        table_dict = azure_api_ocr(
                document_filepath=document_filepath,
                endpoint=self.endpoint,
                subscription_key=self.subscription_key,
                ocr_outputs=None,
                api_type='azure', 
                api='read',
                vertical_columns=VERTICAL_COLUMNS,
                horizontal_columns=HORIZONTAL_COLUMNS,
                horizontal_keywords=HORIZONTAL_KEYWORDS
            )
        
        for k, v in table_dict.items():
            for key, value in v.items():
                for item in VERTICAL_COLUMNS:
                    if value.startswith(item):

             
        
        
        rotated_matrix = rotate_matrix(
            matrix=self.matrix, degree=1, clockwise=True)
        self.assertEqual(self.array_shape(rotated_matrix),
                         self.array_shape(self.matrix))

    def test_rotate_matrix_degree_1_clockwise(self):
        rotated_matrix = rotate_matrix(
            matrix=self.matrix, degree=1, clockwise=True)
        self.assertEqual(rotated_matrix,
                         [['24', '1', '2', '3', '4', '5', '6'],
                          ['23', '40', '25', '26', '27', '28', '7'],
                          ['22', '39', '48', '41', '42', '29', '8'],
                          ['21', '38', '47', '49', '43', '30', '9'],
                          ['20', '37', '46', '45', '44', '31', '10'],
                          ['19', '36', '35', '34', '33', '32', '11'],
                          ['18', '17', '16', '15', '14', '13', '12']])

    def test_rotate_matrix_shape_degree_1_anticlockwise(self):
        rotated_matrix = rotate_matrix(
            matrix=self.matrix, degree=1, clockwise=False)
        self.assertEqual(self.array_shape(rotated_matrix),
                         self.array_shape(self.matrix))

    def test_rotate_matrix_degree_1_anticlockwise(self):
        rotated_matrix = rotate_matrix(
            matrix=self.matrix, degree=1, clockwise=False)
        self.assertEqual(rotated_matrix,
                         [['2', '3', '4', '5', '6', '7', '8'],
                          ['1', '26', '27', '28', '29', '30', '9'],
                          ['24', '25', '42', '43', '44', '31', '10'],
                          ['23', '40', '41', '49', '45', '32', '11'],
                          ['22', '39', '48', '47', '46', '33', '12'],
                          ['21', '38', '37', '36', '35', '34', '13'],
                          ['20', '19', '18', '17', '16', '15', '14']])

    def test_rotate_matrix_shape_any_degree_clockwise(self):
        for degree in self.degrees:
            rotated_matrix = rotate_matrix(
                matrix=self.matrix, degree=degree, clockwise=True)
            self.assertEqual(self.array_shape(rotated_matrix),
                             self.array_shape(self.matrix))

    def test_rotate_matrix_any_degree_clockwise(self):
        for degree in self.degrees:
            rotated_matrix = rotate_matrix(
                matrix=self.matrix, degree=degree, clockwise=True)
            self.assertEqual(
                rotated_matrix,
                self.clockwise_any_degree_result_dictionary[degree])

    def test_rotate_matrix_shape_any_degree_anticlockwise(self):
        for degree in self.degrees:
            rotated_matrix = rotate_matrix(
                matrix=self.matrix, degree=degree, clockwise=False)
            self.assertEqual(self.array_shape(rotated_matrix),
                             self.array_shape(self.matrix))

    def test_rotate_matrix_any_degree_anticlockwise(self):
        for degree in self.degrees:
            rotated_matrix = rotate_matrix(
                matrix=self.matrix, degree=degree, clockwise=False)
            self.assertEqual(
                rotated_matrix,
                self.anticlockwise_any_degree_result_dictionary[degree])

    def test_command_line_interface_options(self):
        target_string = 'Show this message and exit.'
        runner = CliRunner()
        help_result = runner.invoke(matrix_rotation_cli.main, ['--help'])
        self.assertEqual(help_result.exit_code, 0)
        self.assertTrue(target_string in help_result.output)

    def test_command_line_interface_matrix(self):
        input_matrix = "[['1', '2', '3', '4', '5', '6', '7'],\
                         ['24', '25', '26', '27', '28', '29', '8'],\
                         ['23', '40', '41', '42', '43', '30', '9'],\
                         ['22', '39', '48', '49', '44', '31', '10'],\
                         ['21', '38', '47', '46', '45', '32', '11'],\
                         ['20', '37', '36', '35', '34', '33', '12'],\
                         ['19', '18', '17', '16', '15', '14', '13']]"
        target_matrix = "[['24', '1', '2', '3', '4', '5', '6'], ['23', '40', '25', '26', '27', '28', '7'], ['22', '39', '48', '41', '42', '29', '8'], ['21', '38', '47', '49', '43', '30', '9'], ['20', '37', '46', '45', '44', '31', '10'], ['19', '36', '35', '34', '33', '32', '11'], ['18', '17', '16', '15', '14', '13', '12']]"
        runner = CliRunner()
        result = runner.invoke(matrix_rotation_cli.main, args=[
                               "--matrix", input_matrix])
        self.assertEqual(result.exit_code, 0)
        self.assertTrue(target_matrix in result.output)

    def test_command_line_interface_degree(self):
        input_matrix = "[['1', '2', '3', '4', '5', '6', '7'],\
                         ['24', '25', '26', '27', '28', '29', '8'],\
                         ['23', '40', '41', '42', '43', '30', '9'],\
                         ['22', '39', '48', '49', '44', '31', '10'],\
                         ['21', '38', '47', '46', '45', '32', '11'],\
                         ['20', '37', '36', '35', '34', '33', '12'],\
                         ['19', '18', '17', '16', '15', '14', '13']]"
        target_matrix = "[['20', '21', '22', '23', '24', '1', '2'], ['19', '36', '37', '38', '39', '40', '3'], ['18', '35', '44', '45', '46', '25', '4'], ['17', '34', '43', '49', '47', '26', '5'], ['16', '33', '42', '41', '48', '27', '6'], ['15', '32', '31', '30', '29', '28', '7'], ['14', '13', '12', '11', '10', '9', '8']]"
        runner = CliRunner()
        result = runner.invoke(matrix_rotation_cli.main, args=[
                               "--matrix", input_matrix, "--degree", 5])
        self.assertEqual(result.exit_code, 0)
        self.assertTrue(target_matrix in result.output)

    def test_command_line_interface_clockwise(self):
        input_matrix = "[['1', '2', '3', '4', '5', '6', '7'],\
                         ['24', '25', '26', '27', '28', '29', '8'],\
                         ['23', '40', '41', '42', '43', '30', '9'],\
                         ['22', '39', '48', '49', '44', '31', '10'],\
                         ['21', '38', '47', '46', '45', '32', '11'],\
                         ['20', '37', '36', '35', '34', '33', '12'],\
                         ['19', '18', '17', '16', '15', '14', '13']]"
        target_matrix = "[['4', '5', '6', '7', '8', '9', '10'], ['3', '28', '29', '30', '31', '32', '11'], ['2', '27', '44', '45', '46', '33', '12'], ['1', '26', '43', '49', '47', '34', '13'], ['24', '25', '42', '41', '48', '35', '14'], ['23', '40', '39', '38', '37', '36', '15'], ['22', '21', '20', '19', '18', '17', '16']]"
        runner = CliRunner()
        result = runner.invoke(matrix_rotation_cli.main, args=[
                               "--matrix", input_matrix, "--degree", 3,
                               "--clockwise", False])
        self.assertEqual(result.exit_code, 0)
        self.assertTrue(target_matrix in result.output)

    def test_command_line_interface_print_matrix(self):
        input_matrix = "[['1', '2', '3', '4', '5', '6', '7'],\
                         ['24', '25', '26', '27', '28', '29', '8'],\
                         ['23', '40', '41', '42', '43', '30', '9'],\
                         ['22', '39', '48', '49', '44', '31', '10'],\
                         ['21', '38', '47', '46', '45', '32', '11'],\
                         ['20', '37', '36', '35', '34', '33', '12'],\
                         ['19', '18', '17', '16', '15', '14', '13']]"
        target_matrix = """Original Matrix:
[['1', '2', '3', '4', '5', '6', '7'],
 ['24', '25', '26', '27', '28', '29', '8'],
 ['23', '40', '41', '42', '43', '30', '9'],
 ['22', '39', '48', '49', '44', '31', '10'],
 ['21', '38', '47', '46', '45', '32', '11'],
 ['20', '37', '36', '35', '34', '33', '12'],
 ['19', '18', '17', '16', '15', '14', '13']]

Clockwise Rotated Matrix with Degree = 4:
[['21', '22', '23', '24', '1', '2', '3'],
 ['20', '37', '38', '39', '40', '25', '4'],
 ['19', '36', '45', '46', '47', '26', '5'],
 ['18', '35', '44', '49', '48', '27', '6'],
 ['17', '34', '43', '42', '41', '28', '7'],
 ['16', '33', '32', '31', '30', '29', '8'],
 ['15', '14', '13', '12', '11', '10', '9']]
---------------------------------------------
[['21', '22', '23', '24', '1', '2', '3'], ['20', '37', '38', '39', '40', '25', '4'], ['19', '36', '45', '46', '47', '26', '5'], ['18', '35', '44', '49', '48', '27', '6'], ['17', '34', '43', '42', '41', '28', '7'], ['16', '33', '32', '31', '30', '29', '8'], ['15', '14', '13', '12', '11', '10', '9']]"""
        runner = CliRunner()
        result = runner.invoke(matrix_rotation_cli.main, args=[
                               "--matrix", input_matrix, "--degree", 4,
                               "--clockwise", True, "--print_matrix", True])
        self.assertEqual(result.exit_code, 0)
        self.assertTrue(target_matrix in result.output)


if __name__ == '__main__':
    unittest.main()
