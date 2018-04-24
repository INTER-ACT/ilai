# ILAI - Intelligent Language Analyzation and Interpretation

ILAI is a language analyzation tool for text classification and sentiment analysis, built for the INTER!ACT discussion platform.

## Dependencies
Due to the file size limit of GitHub of 100MB, some files are not stored in this repository. They have to be added manually in the [data](master/data) directory:
- **Models Directory:** 
trained ML models.
[Download Link](https://mega.nz/#F!zoQklDiJ!OhdXxQv2A_MFbdvQR6EsJQ)
- **Features Directory:**
files for the feature extraction.
[Download Link](https://mega.nz/#F!r0YGWY7C!eTreeXi1UXsW75l9B0WU0Q)

## Build
1. Download data files
2. Install Python 3.6
3. Install libraries: `pip install -r requirements.txt`
4. Run Django Rest Server: `python manage.py runserver --noreload`

## License
Please note that some libraries used by this project are NOT covered by the following license.

```
Copyright (C) 2017-2018  Markus Leimer
Copyright (C) 2017-2018  Christoph Schopper

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
```
