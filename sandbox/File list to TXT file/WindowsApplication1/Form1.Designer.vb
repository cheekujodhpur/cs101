<Global.Microsoft.VisualBasic.CompilerServices.DesignerGenerated()> _
Partial Class Form1
    Inherits System.Windows.Forms.Form

    'Form overrides dispose to clean up the component list.
    <System.Diagnostics.DebuggerNonUserCode()> _
    Protected Overrides Sub Dispose(ByVal disposing As Boolean)
        Try
            If disposing AndAlso components IsNot Nothing Then
                components.Dispose()
            End If
        Finally
            MyBase.Dispose(disposing)
        End Try
    End Sub

    'Required by the Windows Form Designer
    Private components As System.ComponentModel.IContainer

    'NOTE: The following procedure is required by the Windows Form Designer
    'It can be modified using the Windows Form Designer.  
    'Do not modify it using the code editor.
    <System.Diagnostics.DebuggerStepThrough()> _
    Private Sub InitializeComponent()
        Me.driveList = New System.Windows.Forms.ListBox()
        Me.foldersList = New System.Windows.Forms.ListBox()
        Me.filesList = New System.Windows.Forms.ListBox()
        Me.TextBox1 = New System.Windows.Forms.TextBox()
        Me.Button1 = New System.Windows.Forms.Button()
        Me.SuspendLayout()
        '
        'driveList
        '
        Me.driveList.FormattingEnabled = True
        Me.driveList.Location = New System.Drawing.Point(4, 12)
        Me.driveList.Name = "driveList"
        Me.driveList.Size = New System.Drawing.Size(114, 173)
        Me.driveList.TabIndex = 0
        '
        'foldersList
        '
        Me.foldersList.FormattingEnabled = True
        Me.foldersList.Location = New System.Drawing.Point(124, 3)
        Me.foldersList.Name = "foldersList"
        Me.foldersList.Size = New System.Drawing.Size(148, 199)
        Me.foldersList.TabIndex = 2
        '
        'filesList
        '
        Me.filesList.FormattingEnabled = True
        Me.filesList.Location = New System.Drawing.Point(278, 3)
        Me.filesList.Name = "filesList"
        Me.filesList.Size = New System.Drawing.Size(140, 277)
        Me.filesList.TabIndex = 3
        '
        'TextBox1
        '
        Me.TextBox1.Location = New System.Drawing.Point(476, 24)
        Me.TextBox1.Name = "TextBox1"
        Me.TextBox1.Size = New System.Drawing.Size(226, 20)
        Me.TextBox1.TabIndex = 4
        Me.TextBox1.Text = "C:\Users\bajaj\Desktop\"
        '
        'Button1
        '
        Me.Button1.Location = New System.Drawing.Point(549, 79)
        Me.Button1.Name = "Button1"
        Me.Button1.Size = New System.Drawing.Size(75, 23)
        Me.Button1.TabIndex = 5
        Me.Button1.Text = "save"
        Me.Button1.UseVisualStyleBackColor = True
        '
        'Form1
        '
        Me.AutoScaleDimensions = New System.Drawing.SizeF(6.0!, 13.0!)
        Me.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font
        Me.ClientSize = New System.Drawing.Size(750, 532)
        Me.Controls.Add(Me.Button1)
        Me.Controls.Add(Me.TextBox1)
        Me.Controls.Add(Me.filesList)
        Me.Controls.Add(Me.foldersList)
        Me.Controls.Add(Me.driveList)
        Me.Name = "Form1"
        Me.Text = "file list"
        Me.ResumeLayout(False)
        Me.PerformLayout()

    End Sub
    Friend WithEvents driveList As System.Windows.Forms.ListBox
    Friend WithEvents foldersList As System.Windows.Forms.ListBox
    Friend WithEvents filesList As System.Windows.Forms.ListBox
    Friend WithEvents TextBox1 As System.Windows.Forms.TextBox
    Friend WithEvents Button1 As System.Windows.Forms.Button

End Class
