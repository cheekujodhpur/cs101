Imports System.IO

Public Class Form1

    Private Sub Form1_Load(sender As Object, e As EventArgs) Handles MyBase.Load
        For Each di As DriveInfo In DriveInfo.GetDrives()
            driveList.Items.Add(di)
        Next
    End Sub

    Private Sub ListBox1_SelectedIndexChanged(sender As Object, e As EventArgs) Handles driveList.SelectedIndexChanged
        foldersList.Items.Clear()

        Try

            Dim drive As DriveInfo = DirectCast(driveList.SelectedItem, DriveInfo)

            For Each dirinfo As DirectoryInfo In drive.RootDirectory.GetDirectories()
                foldersList.Items.Add(dirinfo)

            Next
        Catch ex As Exception
            MessageBox.Show(ex.Message)
        End Try
    End Sub

    Private Sub ListBox1_SelectedIndexChanged_1(sender As Object, e As EventArgs) Handles foldersList.SelectedIndexChanged
        filesList.Items.Clear()
        Dim dir As DirectoryInfo = DirectCast(foldersList.SelectedItem, DirectoryInfo)
        For Each fi As FileInfo In dir.GetFiles()
            filesList.Items.Add(fi)
        Next

    End Sub

    Private Sub Button1_Click(sender As Object, e As EventArgs) Handles Button1.Click
        Dim StreamW As New IO.StreamWriter(TextBox1.Text)
        For i = 0 To filesList.Items.Count - 1
            StreamW.WriteLine(filesList.Items.Item(i))
        Next
        StreamW.Close()
        StreamW.Dispose()
        MsgBox("Done")
    End Sub

    Private Sub TextBox1_TextChanged(sender As Object, e As EventArgs) Handles TextBox1.TextChanged

    End Sub

    Private Sub Button2_Click(sender As Object, e As EventArgs)

    End Sub
End Class
