import React, { useState } from 'react';
import './App.css';
import axios from 'axios';
import Button from 'react-bootstrap/Button';

import ImageForm from './ImageForm';
import Image, {NoImage, Loading} from './Image';
import StyleTransferForm from './StyleTransferForm';
import DownloadButton from './DownloadButton';

import 'bootstrap/dist/css/bootstrap.min.css';

const RUN_URL = "http://127.0.0.1:8000/api/run/"
const DOWNLOAD_URL = "http://127.0.0.1:8000/api/download/"

function App() {
    // const [image, setImages] = useState({'image': TestImage});

    // State to be updated when an image is uploaded so that it actually displays
    const [image, setImage] = useState({});
    const [showImage, setShowImage] = useState(false);

    // Same but for the processed output image
    const [output, setOutput] = useState({});
    const [showOutput, setShowOutput] = useState(false);
    const [processing, setProcessing] = useState(false);

    // State for which artist is chosen
    const [weightID, setWeightID] = useState(1);

    // Controls whether button to run neural style transfer is clickable
    const [buttonDisabled, setButtonDisabled] = useState(true);

    function handleImageUpload(data) {
        // console.log(data);
        setImage(data);
        setShowImage(true);
        setButtonDisabled(false);
    }

    function runProcessing(e) {
        // Prevent traditional form submission
        e.preventDefault();

        // Enable processing flag so loading gif can be displayed
        setProcessing(true);

        // Build form for POST data
        let formData = new FormData();
        formData.append('image_id', image.id);
        formData.append('weight_id', weightID);

        // Submit run call with axios
        axios.post(RUN_URL, formData)
            .then((response) => {
                // Turn off loading gif
                setProcessing(false);
                setOutput(response.data);
                setShowOutput(true);
            });
    }

    function downloadResult(e) {
        e.preventDefault();

        // Make GET request to API to save the output image
        let formData = new FormData();
        formData.append('image_id', output.id);
        // formData = {'image_id': output.id};
        axios({
            url: DOWNLOAD_URL,
            method: 'POST',
            responseType: 'blob',
            data: {
                'image_id': output.id
            }
        })
        // axios.post(DOWNLOAD_URL, formData)
            .then((response) => {
                // create file link in browser's memory
                const href = URL.createObjectURL(response.data);

                // create "a" HTML element with href to file & click
                const link = document.createElement('a');
                link.href = href;
                link.setAttribute('download', 'output.jpg');
                document.body.appendChild(link);
                link.click();

                // clean up "a" element & remove ObjectURL
                document.body.removeChild(link);
                URL.revokeObjectURL(href);
            });
    }

    // console.log(image);
    // console.log(setImages);
    // <Image image_response={image} />
    // { showImage ? <Image image_response={image}/> : null }
    // { buttonDisabled ? <Button disabled>Apply Style Transfer</Button> : <Button>Apply Style Transfer</Button>}
    return (
        <>
            <div className="container mt-5 mb-5">
                <ImageForm displayImage={(data) => handleImageUpload(data)}/>
            </div>
            <div className="container mw-50 mh-50">
                { showImage ? <Image image_response={image}/> : <NoImage /> }
            </div>
            <div className="container mt-5 mb-5">
                <StyleTransferForm buttonDisabled={buttonDisabled}
                handleChange={(e) => setWeightID(e.target.value)}
                handleSubmit={runProcessing} />
            </div>
            <div className="container mw-50 mh-50">
                { processing ? <Loading /> : null }
                { showOutput ? <Image image_response={output}/> : null }
                { showOutput ? <DownloadButton buttonText="Download" handleClick={downloadResult}/> : null }
            </div>
        </>
    )
  // return null;
}

export default App;
