var message;
var compiled = []
var totalInput;
var recommendations = [];

function reset() {
  // Clear the compiled array
  compiled = [];

  // Reset the checkbox
  var checkboxes = document.getElementsByClassName("mk");
  for (var i = 0; i < checkboxes.length; i++) {
    checkboxes[i].checked = false;
  }
  
  // Reset the matkul elements to "-"
  var list = document.getElementsByClassName("modal-body")[0];
  var matkulElements = list.getElementsByClassName("matkul");
  for (var i = 0; i < matkulElements.length; i++) {
    matkulElements[i].innerHTML = "-";
  }

  // Reset the nilai elements to empty strings and enable them
  var nilaiElements = list.getElementsByClassName("nilai");
  for (var i = 0; i < nilaiElements.length; i++) {
    nilaiElements[i].value = "";
    nilaiElements[i].removeAttribute("disabled");
  }
  
  // Reset the table cells to empty values
  var table = document.getElementsByClassName("modal-body")[1];
  var tbody = table.getElementsByTagName("tbody")[0];
  for (var i = 0; i < 5; i++) {
    var row = tbody.rows[i];
    row.cells[0].innerHTML = "";
    row.cells[1].innerHTML = "";
    row.cells[2].innerHTML = "";
  }

  // Hide the modal
  var modal = document.getElementsByClassName("modal fade rekomendasi")[0];
  modal.style.display = "none";
}

function getData() {
  var inputElements = document.getElementsByClassName('mk');
  for (var i = 0; inputElements[i]; i++) {
    if (inputElements[i].checked) {
      compiled.push(inputElements[i].value);
    }
  }

  console.log(compiled)

  var list = document.getElementsByClassName("modal-body")[0];
  //mendapatkan seluruh mata kuliah yang sudah dipilih
  for (var i = 0; i < compiled.length; i++) {
    list.getElementsByClassName("matkul")[i].innerHTML = compiled[i];
    totalInput = i + 1
  }
  //mengganti paragraph dari matkul ke i menjadi - dan mendisable text area
  for (var i = totalInput; i < 7; i++) {
    list.getElementsByClassName("matkul")[i].innerHTML = "-"
    list.getElementsByClassName("nilai")[i].disabled = true;
  }
}

function getNilai() {
  var inputElements = document.getElementsByClassName("nilai");

  for (var i = 0; inputElements[i]; i++) {
    var selectedValue = inputElements[i].value;
    if (selectedValue && selectedValue != "none") {
      compiled[i] = {
        "matkul": compiled[i],
        "nilai": parseFloat(selectedValue)
      }
    }
  }
  console.log(compiled)

  //mengirim data ke back-end untuk diolah
  calltoServer(compiled);
  //compiled = []
}

function calltoServer(value) {
  var xhttp = new XMLHttpRequest();

  xhttp.onreadystatechange = function() {
    if (this.readyState == XMLHttpRequest.DONE) {
      var response = JSON.parse(this.responseText);
      recommendations = response.recommendations;

      console.log(recommendations)

      // Select the table element to populate with the recommendations
      var table = document.getElementsByClassName("modal-body")[1];
      
      // Select the tbody element of the table
      var tbody = table.getElementsByTagName("tbody")[0];

      for (var i = 0; i < 5; i++) {
          var row = tbody.rows[i];
          row.cells[0].innerHTML = recommendations[i].Dosen;
          row.cells[1].innerHTML = recommendations[i].Quota;
          row.cells[2].innerHTML = recommendations[i].Title;
      }

      // Select the keyword element to populate with unique keywords
      var keywordElement = document.getElementsByClassName("Keyword")[0];

      // Create an empty array to store unique keywords
      var uniqueKeywords = [];

      // Loop through the recommendations to extract the keywords
      for (var i = 0; i < recommendations.length; i++) {
        var keywords = recommendations[i].Keyword.split(",");
        for (var j = 0; j < keywords.length; j++) {
          var keyword = keywords[j].trim();
          if (!uniqueKeywords.includes(keyword)) {
            uniqueKeywords.push(keyword);
            var keywordSpan = document.createElement("span");
            keywordSpan.innerHTML = keyword;
            keywordElement.appendChild(keywordSpan);
          }
        }
      }

      // Add spaces between keywords
      var formattedKeywords = keywords.join(", ");

      keywordElement.innerHTML = formattedKeywords;

      var modal = document.getElementsByClassName("modal fade rekomendasi")[0];
      modal.style.display = "block";
    }
  };

  xhttp.open("POST", "/result_byacademicprofile", true);
  xhttp.setRequestHeader("Content-Type", "application/json");
  xhttp.send(JSON.stringify(value));
}



function checkcontrol(j) {
  var totalchecked = 0;
  for (var i = 0; i < document.formMataKuliah.rGroup.length; i++) {
    if (document.formMataKuliah.rGroup[i].checked) {
      totalchecked = totalchecked + 1;
    }
    if (totalchecked > 7) {
      alert("You can only choose 7 courses")
      document.formMataKuliah.rGroup[j].checked = false;
      return false;
    }
  }
}
