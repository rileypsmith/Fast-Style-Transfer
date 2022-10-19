import React from 'react';

import Form from 'react-bootstrap/Form';
import Button from 'react-bootstrap/Button';

export default function StyleTransferForm({buttonDisabled, handleChange, handleSubmit}) {
    return (
        <div className="d-grid gap-2">
            <Form.Select onChange={handleChange}>
                <option value="1">Van Goh</option>
                <option value="2">Hokusai</option>
                <option value="3">Cezanne</option>
                <option value="4">Wang</option>
                <option value="5">Pissarro</option>
                <option value="6">Okeefe</option>
                <option value="7">Monet</option>
            </Form.Select>
            <Button disabled={buttonDisabled}
            onClick={handleSubmit} variant="secondary">Apply Style Transfer</Button>
        </div>
    )
}
