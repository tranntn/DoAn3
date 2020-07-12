// Accord.NET Sample Applications
// http://accord-framework.net
//
// Copyright © 2009-2017, César Souza
// All rights reserved. 3-BSD License:
//
//   Redistribution and use in source and binary forms, with or without
//   modification, are permitted provided that the following conditions are met:
//
//      * Redistributions of source code must retain the above copyright
//        notice, this list of conditions and the following disclaimer.
//
//      * Redistributions in binary form must reproduce the above copyright
//        notice, this list of conditions and the following disclaimer in the
//        documentation and/or other materials provided with the distribution.
//
//      * Neither the name of the Accord.NET Framework authors nor the
//        names of its contributors may be used to endorse or promote products
//        derived from this software without specific prior written permission.
// 
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
//  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
//  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//  DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
//  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
//  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
//  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
//  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
//  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// 

using Accord;
using Accord.IO;
using Accord.MachineLearning.DecisionTrees;
using Accord.MachineLearning.DecisionTrees.Learning;
using Accord.Math;
using Accord.Statistics.Analysis;
using AForge;
using Components;
using System;
using System.Data;
using System.Drawing;
using System.IO;
using System.Windows.Forms;
using ZedGraph;

namespace SampleApp
{
    /// <summary>
    ///   Classification using Decision Trees.
    /// </summary>
    /// 
    public partial class MainForm : Form
    {

        // current tree
        DecisionTree tree;

        // source column names
        string[] columnNames;



        public MainForm()
        {
            InitializeComponent();

            dgvLearningSource.AutoGenerateColumns = true;
            dgvPerformance.AutoGenerateColumns = false;

            openFileDialog.InitialDirectory = Path.Combine(Application.StartupPath, "Resources");
        }



        /// <summary>
        ///   Creates and learns a Decision Tree to recognize the
        ///   previously loaded dataset using the current settings.
        /// </summary>
        /// 
        private void btnCreate_Click(object sender, EventArgs e)
        {
            if (dgvLearningSource.DataSource == null)
            {
                MessageBox.Show("Please load some data first.");
                return;
            }

            // Finishes and save any pending changes to the given data
            dgvLearningSource.EndEdit();

            // Creates a matrix from the entire source data table
            double[,] table = (dgvLearningSource.DataSource as DataTable).ToMatrix(out columnNames);

            // Get only the input vector values (first two columns)
            double[][] inputs = table.GetColumns( 0, 1, 2, 3, 4, 5).ToJagged();

            // Get the expected output labels (last column)
            int[] outputs = table.GetColumn(6).ToInt32();

            // Specify the input variables
            DecisionVariable[] variables =
            {
                new DecisionVariable("Pclass", DecisionVariableKind.Continuous),
                new DecisionVariable("Sex", DecisionVariableKind.Continuous),
                new DecisionVariable("Parch", DecisionVariableKind.Continuous),
                new DecisionVariable("Fare", DecisionVariableKind.Continuous),
                new DecisionVariable("Age", DecisionVariableKind.Continuous),
                new DecisionVariable("Embarked", DecisionVariableKind.Continuous),
            };

            // Create the C4.5 learning algorithm
            var c45 = new C45Learning(variables);
            // Learn the decision tree using C4.5
            tree = c45.Learn(inputs, outputs);

            // Show the learned tree in the view
            decisionTreeView1.TreeSource = tree;

            // Get the ranges for each variable (X and Y)
            DoubleRange[] ranges = table.GetRange(0);

            CreateScatterplot(zedGraphControl2, table);
            lbStatus.Text = "Learning finished! Click the other tabs to explore results!";
        }


        /// <summary>
        ///   Tests the previously created tree into a new set of data.
        /// </summary>
        /// 
        private void btnTestingRun_Click(object sender, EventArgs e)
        {
            if (tree == null || dgvTestingSource.DataSource == null)
            {
                MessageBox.Show("Please create a machine first.");
                return;
            }
            // Creates a matrix from the entire source data table
            double[][] table = (dgvLearningSource.DataSource as DataTable).ToJagged(out columnNames);
            // Get only the input vector values (first two columns)
            double[][] inputs = table.GetColumns(0, 1 , 2, 3, 4, 5);
            // Get the expected output labels (last column)
            int[] expected = table.GetColumn(6).ToInt32();
            // Compute the actual tree outputs
            int[] actual = tree.Decide(inputs);
            // Use confusion matrix to compute some statistics.
            ConfusionMatrix confusionMatrix = new ConfusionMatrix(actual, expected, 1, 0);
            dgvPerformance.DataSource = new[] { confusionMatrix };
            // Create performance scatter plot
            CreateResultScatterplot(zedGraphControl1, inputs, expected.ToDouble(), actual.ToDouble());
        }






        private void MenuFileOpen_Click(object sender, EventArgs e)
        {
            if (openFileDialog.ShowDialog(this) == DialogResult.OK)
            {
                string filename = openFileDialog.FileName;
                string extension = Path.GetExtension(filename);
                if (extension == ".xls" || extension == ".xlsx")
                {
                    ExcelReader db = new ExcelReader(filename, false, false);
                    TableSelectDialog t = new TableSelectDialog(db.GetWorksheetList());

                    if (t.ShowDialog(this) == DialogResult.OK)
                    {
                        DataTable tableSource = db.GetWorksheet(t.Selection);

                        double[,] sourceMatrix = tableSource.ToMatrix(out columnNames);

                        // Detect the kind of problem loaded.
                        if (sourceMatrix.GetLength(1) == 2)
                        {
                            MessageBox.Show("Missing class column.");
                        }
                        else
                        {
                            this.dgvLearningSource.DataSource = tableSource;
                            this.dgvTestingSource.DataSource = tableSource.Copy();


                            CreateScatterplot(graphInput, sourceMatrix);
                        }
                    }
                }
            }

            lbStatus.Text = "When ready, click 'Create Tree' to start the tree inducing algorithm!";
        }


        public void CreateScatterplot(ZedGraphControl zgc, double[,] graph)
        {
            GraphPane myPane = zgc.GraphPane;
            myPane.CurveList.Clear();

            // Set the titles
            myPane.Title.Text = "Decision Tree Titanic";
            myPane.XAxis.Title.Text = "Age";
            myPane.YAxis.Title.Text = "Fare";


            // Classification problem
            PointPairList listNotSurvived = new PointPairList(); // Z = 0
            PointPairList listSurVived = new PointPairList(); // Z = 1
            for (int i = 0; i < graph.GetLength(0); i++)
            {
                if (graph[i, 6] == 0)
                    listNotSurvived.Add(graph[i, 2], graph[i, 4]);
                if (graph[i, 6] == 1)
                    listSurVived.Add(graph[i, 2], graph[i, 4]);
            }

            // Add the curve
            LineItem myCurve = myPane.AddCurve("Not Survived", listNotSurvived, Color.Blue, SymbolType.Diamond);
            myCurve.Line.IsVisible = false;
            myCurve.Symbol.Border.IsVisible = false;
            myCurve.Symbol.Fill = new Fill(Color.Blue);

            myCurve = myPane.AddCurve("Survived", listSurVived, Color.Green, SymbolType.Diamond);
            myCurve.Line.IsVisible = false;
            myCurve.Symbol.Border.IsVisible = false;
            myCurve.Symbol.Fill = new Fill(Color.Green);


            // Fill the background of the chart rect and pane
            //myPane.Chart.Fill = new Fill(Color.White, Color.LightGoldenrodYellow, 45.0f);
            //myPane.Fill = new Fill(Color.White, Color.SlateGray, 45.0f);
            //myPane.Fill = new Fill(Color.WhiteSmoke);

            zgc.AxisChange();
            zgc.Invalidate();
        }


        public void CreateResultScatterplot(ZedGraphControl zgc, double[][] inputs, double[] expected, double[] output)
        {
            GraphPane myPane = zgc.GraphPane;
            myPane.CurveList.Clear();

            // Set the titles
            myPane.Title.Text = "Decision Tree Titanic";
            myPane.XAxis.Title.Text = "Age";
            myPane.YAxis.Title.Text = "Fare";


            // Classification problem
            PointPairList list1 = new PointPairList(); // Z = 0, OK
            PointPairList list2 = new PointPairList(); // Z = 1, OK
            PointPairList list3 = new PointPairList(); // Z = 0, Error
            PointPairList list4 = new PointPairList(); // Z = 1, Error
            for (int i = 0; i < output.Length; i++)
            {
                if (output[i] == 0)
                {
                    if (expected[i] == 0)
                        list1.Add(inputs[i][2], inputs[i][4]);
                    if (expected[i] == 1)
                        list3.Add(inputs[i][2], inputs[i][4]);
                }
                else
                {
                    if (expected[i] == 0)
                        list4.Add(inputs[i][2], inputs[i][4]);
                    if (expected[i] == 1)
                        list2.Add(inputs[i][2], inputs[i][4]);
                }
            }

            // Add the curve
            LineItem
            myCurve = myPane.AddCurve("Non Survived Hits", list1, Color.Blue, SymbolType.Diamond);
            myCurve.Line.IsVisible = false;
            myCurve.Symbol.Border.IsVisible = false;
            myCurve.Symbol.Fill = new Fill(Color.Blue);

            myCurve = myPane.AddCurve("Survived Hits", list2, Color.Green, SymbolType.Diamond);
            myCurve.Line.IsVisible = false;
            myCurve.Symbol.Border.IsVisible = false;
            myCurve.Symbol.Fill = new Fill(Color.Green);

            myCurve = myPane.AddCurve("Non Survived Miss", list3, Color.Blue, SymbolType.Plus);
            myCurve.Line.IsVisible = false;
            myCurve.Symbol.Border.IsVisible = true;
            myCurve.Symbol.Fill = new Fill(Color.Blue);

            myCurve = myPane.AddCurve("Survived Miss", list4, Color.Green, SymbolType.Plus);
            myCurve.Line.IsVisible = false;
            myCurve.Symbol.Border.IsVisible = true;
            myCurve.Symbol.Fill = new Fill(Color.Green);


            // Fill the chart panel background color
            myPane.Fill = new Fill(Color.WhiteSmoke);

            zgc.AxisChange();
            zgc.Invalidate();
        }

        private void toolStripMenuItem7_Click(object sender, EventArgs e)
        {
            new AboutBox().ShowDialog(this);
        }

        private void menuStrip1_ItemClicked(object sender, ToolStripItemClickedEventArgs e)
        {

        }

        private void toolStripMenuItem1_Click(object sender, EventArgs e)
        {

        }

        private void toolStripSeparator3_Click(object sender, EventArgs e)
        {

        }

        private void toolStripMenuItem5_Click(object sender, EventArgs e)
        {

        }

        private void toolStripMenuItem6_Click(object sender, EventArgs e)
        {

        }

        private void openFileDialog_FileOk(object sender, System.ComponentModel.CancelEventArgs e)
        {

        }

        private void tabPage4_Click(object sender, EventArgs e)
        {

        }

        private void splitContainer1_SplitterMoved(object sender, SplitterEventArgs e)
        {

        }

        private void groupBox2_Enter(object sender, EventArgs e)
        {

        }

        private void dgvTestingSource_CellContentClick(object sender, DataGridViewCellEventArgs e)
        {

        }

        private void groupBox11_Enter(object sender, EventArgs e)
        {

        }

        private void zedGraphControl1_Load(object sender, EventArgs e)
        {

        }

        private void groupBox6_Enter(object sender, EventArgs e)
        {

        }

        private void dgvPerformance_CellContentClick(object sender, DataGridViewCellEventArgs e)
        {

        }

        private void tabOverview_Click(object sender, EventArgs e)
        {

        }

        private void splitContainer2_SplitterMoved(object sender, SplitterEventArgs e)
        {

        }

        private void groupBox1_Enter(object sender, EventArgs e)
        {

        }

        private void decisionTreeView1_Load(object sender, EventArgs e)
        {

        }

        private void groupBox3_Enter(object sender, EventArgs e)
        {

        }

        private void zedGraphControl2_Load(object sender, EventArgs e)
        {

        }

        private void tabSamples_Click(object sender, EventArgs e)
        {

        }

        private void splitContainer7_SplitterMoved(object sender, SplitterEventArgs e)
        {

        }

        private void groupBox7_Enter(object sender, EventArgs e)
        {

        }

        private void dgvLearningSource_CellContentClick(object sender, DataGridViewCellEventArgs e)
        {

        }

        private void groupBox15_Enter(object sender, EventArgs e)
        {

        }

        private void graphInput_Load(object sender, EventArgs e)
        {

        }

        private void tabControl_SelectedIndexChanged(object sender, EventArgs e)
        {

        }

        private void statusStrip1_ItemClicked(object sender, ToolStripItemClickedEventArgs e)
        {

        }

        private void lbStatus_Click(object sender, EventArgs e)
        {

        }
    }
}
