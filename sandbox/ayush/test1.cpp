/*
	This program detects faces in continuous video stream from webcam and shows the cropped faces,
	
	also storing them in consecutively numbered files
	
	Copyright (C) <2014> <Group 02: Kumar Ayush, Reebhu Bhattacharyya, Kshitij Bajaj, Keshav Srinivasan>

	This program is free software: you can redistribute it and/or modify

	it under the terms of the GNU General Public License as published by

	the Free Software Foundation, either version 3 of the License, or

	(at your option) any later version.

	This program is distributed in the hope that it will be useful,

	but WITHOUT ANY WARRANTY; without even the implied warranty of

	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the

	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License

	along with this program. If not, see <http://www.gnu.org/licenses/>.

*/
#include "Header.h"
#include<iostream>
using namespace std;
int main(char* argv[], int argc)
{
	
	cout << "videoFaceDetect.cpp Copyright(C) <2014> <Group 02: Kumar Ayush, Reebhu Bhattacharyya, Kshitij Bajaj, Keshav Srinivasan>\n\n";
	cout << "This program comes with ABSOLUTELY NO WARRANTY.\n\n";
	cout << "This is free software, and you are welcome to redistribute it\n\n";
	cout << "under certain conditions.";
	cout << "Press Esc to stop streaming.";
	videoFaceDetect();//call a function to open videostream(webcam), detect faces in it and show cropped faces
	return 0;
}

