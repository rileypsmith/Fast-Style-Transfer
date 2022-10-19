import React, { useState } from 'react';
import axios from 'axios';
import FileUploader from './FileUploader';
import Form from "react-bootstrap/Form";

const UPLOAD_URL = "http://127.0.0.1:8000/api/create/"

export default function ImageForm({ displayImage }) {

    // State to be updated for which image is being uploaded
    const [selectedFile, setSelectedFile] = useState("");

    // Function to upload and display an image when it is chosen from file select
    // This allows uploading to take place without clicking a separate "upload" button
    function handleUpload(file) {
        // Make FormData object and populate with selected image
        let formData = new FormData();
        formData.append("image", file);

        // Post it using axios
        axios.post(UPLOAD_URL, formData)
            .then(function(response) {
                displayImage(response.data);
                // console.log('res: ', res);
                // alert("File upload success");
            });
    }

    // const submitForm = (e) => {
    //     // Prevent automatic page refresh
    //     e.preventDefault();
    //
    //     // Make FormData object and populate with selected image
    //     let formData = new FormData();
    //     formData.append("image", selectedFile);
    //
    //     // Post it using axios
    //     axios.post(UPLOAD_URL, formData)
    //         .then(function(res) {
    //             console.log('res: ', res);
    //             alert("File upload success");
    //         });
    // };

    // <button onClick={submitForm}>Upload</button>
    return(
        <Form>
            <FileUploader onFileSelect={(file) => handleUpload(file)} />
        </Form>
    )
}
