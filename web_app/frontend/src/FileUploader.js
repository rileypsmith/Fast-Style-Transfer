import React from 'react';
import Form from "react-bootstrap/Form";

export default function FileUploader({onFileSelect}) {

    return (
        <div className="file-uploader">
            <Form.Control type="file" onChange={(e) => onFileSelect(e.target.files[0])} />
        </div>
    )

}
